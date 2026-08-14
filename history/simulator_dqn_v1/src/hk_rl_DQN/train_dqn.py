"""训练并评估 Boss 战斗环境中的 Double DQN 智能体。

本模块包含完整学习链路：归一化观测、把交互存入经验回放、计算
"""
    
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# 通常从仓库根目录通过 ``python -m hk_rl_DQN.train_dqn`` 启动本包。
# 下面的回退导入也允许用户已在当前目录时，使用更短的
# ``python -m train_dqn`` 命令启动。
try:
    from .boss_env import BossDodgeEnv
except ImportError:  # 支持从当前目录执行 `python -m train_dqn`。
    from boss_env import BossDodgeEnv


# 网络的一次决策连续控制两个模拟帧。这样既缩短了有效决策时域，
# 又避免策略以不符合实际操作习惯的超高频率来回改变方向。
ACTION_REPEAT = 2

# 环境每个模拟帧都会产生奖励，因此重复执行的帧仍需逐帧折扣。
# GAMMA 是经验回放中相邻两次“决策”之间的等效折扣率。
FRAME_GAMMA = 0.995
GAMMA = FRAME_GAMMA ** ACTION_REPEAT

# 时间上限同时约束单局内存使用和最坏训练耗时。状态由环境完整的
# 18 个数值观测组成；不同于旧 Q 表，DQN 可直接接收连续值，
# 无需先把它们离散化成有限格子。
MAX_EPISODE_STEPS = 1200
STATE_DIMENSIONS = 18
STATE_ENCODING = "normalized-observation-v1"

# 路径相对于启动 Python 时的工作目录。指标与权重分开保存后，
# 无需反序列化 PyTorch 对象也能直接读取并绘制学习曲线。
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/dqn.pt")
DEFAULT_METRICS_PATH = Path("runs/dqn.json")

# 对自举目标使用较小学习率的 AdamW，是较保守稳定的设置。
# 批量大小 128 能获得有代表性的平均梯度，对这里的两层网络而言
# 计算规模仍然很小。
LEARNING_RATE = 1e-4
BATCH_SIZE = 128

# 连续游戏帧高度相关；经验回放通过随机抽取过去交互来打乱这种顺序。
# 预热阶段会延迟梯度更新，直到缓冲区至少积累了一批具有基本多样性的局面。
REPLAY_CAPACITY = 50_000
REPLAY_WARMUP = 1_000

# 目标网络有意在 500 次决策内保持不变。如果它持续紧跟在线网络，
# Bellman 等式两端会同时移动，学习过程更容易振荡甚至发散。
TARGET_UPDATE_INTERVAL = 500

# Epsilon 表示随机选择一个合法动作的概率。探索率从 100% 线性下降到 5%；
# 保留这个小下限，可让智能体在早期策略形成后仍有机会发现替代方案。
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 40_000

# 稀有的终局奖励可能产生很大的时序差分误差。Huber loss 与梯度裁剪
# 从损失和梯度两个层面共同限制此类极端更新。
GRADIENT_CLIP_NORM = 10.0
HIDDEN_DIMENSIONS = (128, 128)

# 动作重复需要合并两帧的 ``info`` 字典。以下事件字段表示“任一帧发生过”，
# 所以应使用逻辑 OR 聚合；如果只保留最后一帧，可能悄悄丢失命中或闪避事件。
REPEATED_EVENT_INFO_KEYS = (
    "boss_hit",
    "boss_teleported",
    "spike_dodged",
    "entered_attack_range",
    "spike_escape_timeout",
)


# 阅读路线：先看状态如何缩放，再看合法动作如何筛选，随后看经验回放。
# 真正的 DQN 数学集中在 ``optimize_model``；``train`` 负责把环境交互、
# 回放采样和网络更新串成完整流程。第一次阅读不必先钻进命令行部分。
def encode_state(observation: Sequence[float | int]) -> tuple[float, ...]:
    """将环境观测归一化为适合神经网络的输入。

    当输入特征数量级接近时，神经网络训练通常更稳定。位置除以场地尺寸，
    计时器除以各自的自然持续时间，而带符号方向本来就在 ``[-1, 1]`` 附近。
    """
    # 在这里尽早失败很重要：如果环境改变了观测结构，旧缩放列表与新观测
    # 直接 zip 会静默截断并丢失特征，错误将很难追踪。
    if len(observation) != STATE_DIMENSIONS:
        raise ValueError(
            f"expected {STATE_DIMENSIONS} observation values, got {len(observation)}"
        )
    # 统一转换为浮点数，既便于后续构造张量，也避免不同经验之间发生
    # 数据类型变化。
    values = tuple(float(value) for value in observation)

    # 顺序与 ``BossDodgeEnv._observation`` 完全一致：玩家运动与冷却、
    # Boss 运动、攻击状态，最后是生命值和无敌状态。每个分母都对应
    # 该特征直观合理的最大尺度。
    scales = (
        # 玩家位置：相对于整个场地宽高。
        BossDodgeEnv.ARENA_WIDTH,
        BossDodgeEnv.ARENA_HEIGHT,
        # 玩家垂直速度与朝向：速度按最大下落速度缩放。
        abs(BossDodgeEnv.TERMINAL_VELOCITY),
        1.0,
        # 攻击与冲刺状态：计时器按各自最长冷却或持续时间缩放。
        BossDodgeEnv.PLAYER_ATTACK_RECOVERY_FRAMES,
        BossDodgeEnv.DASH_FRAMES,
        1.0,
        BossDodgeEnv.PLAYER_DASH_RECOVERY_FRAMES,
        # Boss 位置与水平速度：位置仍相对于场地，速度按基础速度缩放。
        BossDodgeEnv.ARENA_WIDTH,
        BossDodgeEnv.ARENA_HEIGHT,
        BossDodgeEnv.BOSS_SPEED,
        # 地刺判定框位置：不存在时会在下方单独恢复成 -1 哨兵值。
        BossDodgeEnv.ARENA_WIDTH,
        BossDodgeEnv.ARENA_HEIGHT,
        # 攻击阶段和计时器：分别按最大阶段编号和主动攻击帧数缩放。
        BossDodgeEnv.ATTACK_RECOVERY,
        BossDodgeEnv.ATTACK_ACTIVE_FRAMES,
        # 双方生命值以及玩家受伤后的无敌计时器。
        BossDodgeEnv.INITIAL_PLAYER_HP,
        BossDodgeEnv.INITIAL_BOSS_HP,
        BossDodgeEnv.HURT_INVULNERABILITY_FRAMES,
    )
    # 这里有意使用确定性且无状态的归一化。若采用运行均值和方差，
    # 每个检查点都必须额外保存统计量，并可能在训练与可视化之间发生漂移。
    normalized = [value / scale for value, scale in zip(values, scales)]

    # 环境用 -1 表示攻击判定框不存在。这里把它保留为明确且有界的哨兵值，
    # 而不是缩放成一个很小的数，否则网络可能误认为攻击位于左侧或底部边缘。
    normalized[11] = -1.0 if values[11] < 0 else normalized[11]
    normalized[12] = -1.0 if values[12] < 0 else normalized[12]

    # 元组不可变且适合作为紧凑的回放记录，可防止环境继续推进时
    # 意外原地修改已经存入经验池的状态。
    return tuple(normalized)


# 直观理解：网络可以给每个动作打分，但不是每个动作此刻都能执行。
# 这里像一层“规则过滤器”，避免冷却中的攻击、空中二次跳跃等动作
# 参与探索或最大值比较，让网络专注学习真正有意义的选择。
def available_action_indices(env: BossDodgeEnv) -> tuple[int, ...]:
    """返回当前帧真正能够产生作用的动作。

    屏蔽不可能动作可让探索更有效：冷却期间随机攻击不会提供新信息。
    同时也能防止某个无效动作因 Q 值被高估而赢得贪心选择。
    """
    # 先假设所有动作可用，再移除前置条件不满足的动作。``wait`` 永不移除，
    # 因而始终至少存在一个合法动作。
    available = set(range(len(env.ACTIONS)))

    # 攻击与冲刺拥有独立冷却；任一冷却期间仍允许移动，
    # 与环境真实控制规则保持一致。
    if env.player_attack_recovery_timer > 0:
        available.discard(env.ACTIONS.index("attack"))
    if env.player_dash_recovery_timer > 0:
        available.discard(env.ACTIONS.index("dash"))

    # 玩家没有二段跳；朝场地墙壁继续移动也会被排除，因为其结果
    # 与在墙边等待完全相同。
    if not env.is_grounded:
        available.discard(env.ACTIONS.index("jump"))
    if env.player_x <= 0:
        available.discard(env.ACTIONS.index("left"))
    if env.player_x >= env.ARENA_WIDTH - env.PLAYER_WIDTH:
        available.discard(env.ACTIONS.index("right"))
    # 排序让动作顺序在不同 Python 运行与随机种子下保持确定。
    return tuple(sorted(available))


def action_mask(env: BossDodgeEnv) -> tuple[bool, ...]:
    """把下一状态的动作可用性表示为固定宽度的掩码。

    索引元组适合选择单个动作，布尔行则适合批量屏蔽所有 Q 值。
    合法性属于记录下来的下一状态，而不是不断变化的实时环境，
    因此经验回放会连同状态一起保存此掩码。
    """
    available = set(available_action_indices(env))
    return tuple(index in available for index in range(len(env.ACTIONS)))


def select_greedy_action(
    action_values: Sequence[float] | Tensor,
    rng: random.Random,
    available_actions: tuple[int, ...] | None = None,
) -> int:
    """在并列最大值的合法动作中均匀随机选择。

    训练早期所有输出值往往很接近，因此随机打破平局十分重要。
    若总取第一个最大值，会无意中偏向 ``BossDodgeEnv.ACTIONS`` 的首项。
    """
    # 动作选择本身不可微，因此把这个小向量移到 CPU 并转换为普通数值，
    # 不会破坏学习所需的梯度路径。
    values = (
        action_values.detach().cpu().tolist()
        if isinstance(action_values, Tensor)
        else list(action_values)
    )

    # 显式校验可在问题源头给出清晰错误，而不是稍后由 ``max`` 或
    # ``random.choice`` 抛出难以理解的异常。
    if not values:
        raise ValueError("action_values must not be empty")
    available = available_actions or tuple(range(len(values)))
    if not available:
        raise ValueError("available_actions must not be empty")
    # 最大值计算和平局集合都只包含合法动作。某个非法动作即使有很高的
    # 原始网络输出也无法执行，因此绝不能影响最终选择。
    best_value = max(values[index] for index in available)
    best = [index for index in available if values[index] == best_value]
    return rng.choice(best)


# 直观理解：智能体按“决策步”思考，环境按“画面帧”运行。
# 一个决策连续执行两帧后才形成一条经验，因此这里必须合并两帧奖励、
# 事件和终止信号，保证回放中的一条数据确实对应一次网络决策。
def step_with_action_repeat(
    env: BossDodgeEnv,
    action: int,
) -> tuple[tuple[float | int, ...], float, bool, bool, dict[str, object]]:
    """重复执行一次决策，并聚合奖励、终止信号和事件。

    从 DQN 视角看，整个函数只产生一条 transition。其奖励是底层各帧
    奖励的折扣和，返回观测则是执行一帧或两帧后到达的状态。
    """
    # 标量奖励跨帧相加，布尔事件用 OR 聚合；普通 info 字段取最终到达帧，
    # 因为该帧正是返回状态所对应的时刻。
    total_reward = 0.0
    repeated_events = {key: False for key in REPEATED_EVENT_INFO_KEYS}
    progress_penalty = 0.0
    observation: tuple[float | int, ...] | None = None
    terminated = truncated = False
    info: dict[str, object] = {}
    frames_advanced = 0
    # 遇到自然终止或时间截断时立即停止重复。终局后再次调用 ``env.step``
    # 会产生无效样本。
    for repeat_index in range(ACTION_REPEAT):
        observation, reward, terminated, truncated, frame_info = env.step(action)
        # 第 0 帧权重为 1，第 1 帧权重为 FRAME_GAMMA；后续 Bellman 更新
        # 再用 GAMMA 从本次决策折扣到下一次决策。
        total_reward += (FRAME_GAMMA**repeat_index) * reward
        frames_advanced += 1
        info = dict(frame_info)
        # 即使短暂事件只发生在重复动作的第一帧，也必须将它保留下来。
        for key in repeated_events:
            repeated_events[key] = repeated_events[key] or bool(frame_info[key])
        progress_penalty += float(frame_info["progress_penalty"])
        if terminated or truncated:
            break
    # ACTION_REPEAT 当前为正数；此防御检查主要用于避免未来修改配置时
    # 让函数在未推进环境的情况下继续执行。
    if observation is None:
        raise RuntimeError("action repeat did not advance the environment")
    # 向指标和测试暴露聚合细节。``frames_advanced`` 正常等于 ACTION_REPEAT，
    # 若提前终止则可能只有 1。
    info.update(repeated_events)
    info["progress_penalty"] = progress_penalty
    info["frames_advanced"] = frames_advanced
    return observation, total_reward, terminated, truncated, info


# 网络并不直接输出动作，而是输出六个“长期收益估计”。
# 数值越大表示从当前状态执行对应动作后，未来累计奖励预计越高；
# epsilon-greedy 再根据这些估计决定利用已有知识还是随机探索。
class DQN(nn.Module):
    """把归一化状态映射为每个离散动作的预计回报。

    输出层有意保持线性：Q 值是无界的折扣回报而非概率。使用 sigmoid
    或 softmax 会赋予错误含义，并限制负值或足够大的预测值。
    """

    def __init__(
        self,
        state_dimensions: int = STATE_DIMENSIONS,
        action_count: int = len(BossDodgeEnv.ACTIONS),
        hidden_dimensions: Sequence[int] = HIDDEN_DIMENSIONS,
    ) -> None:
        super().__init__()
        # 用维度元组构建网络，既保留默认的 18 -> 128 -> 128 -> 6 结构，
        # 又能让检查点按记录的维度通用地重建网络。
        dimensions = (state_dimensions, *hidden_dimensions, action_count)
        layers: list[nn.Module] = []

        # 隐藏线性层后接 ReLU，输出层后不接激活函数。最终六个数的位置
        # 与环境动作元组逐项对应。
        for layer_index, (input_size, output_size) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layers.append(nn.Linear(input_size, output_size))
            if layer_index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        # 模型只有一条前馈路径，不需要自定义循环状态或分支，
        # 因此 Sequential 已足够表达整个结构。
        self.network = nn.Sequential(*layers)

    def forward(self, states: Tensor) -> Tensor:
        # ``states`` 可以是单个状态或一批状态；Linear 会保留所有前导维度，
        # 只替换最后一个特征维度。
        return self.network(states)


# 一条经验可读作：在 state 做 action，得到 reward，到达 next_state。
# done 表示故事在这里结束；next_action_mask 则记录下一状态能做什么，
# 避免以后抽样时错误地读取已经变化的实时环境状态。
@dataclass(frozen=True)
class Transition:
    """Bellman 更新所使用的一条决策级经验。

    ``done`` 阻止价值越过 episode 边界继续自举；next mask 记录观测到
    ``next_state`` 时哪些动作合法。
    """

    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    next_action_mask: tuple[bool, ...]


# 为什么不只用刚发生的一步训练：相邻帧过于相似，连续学习容易偏向
# 最近的一种局面。回放池把过去经验打乱抽样，让每个 batch 更接近
# 多种战斗局面的混合，从而降低梯度相关性并提高数据利用率。
class ReplayBuffer:
    """固定容量、先进先出的历史交互存储器。

    deque 填满后会自动丢弃最旧经验，从而限制内存使用，并随着当前策略
    改善，逐渐把训练数据分布转向较新的行为。
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        # 非正容量永远无法提供训练 batch，因此在构造阶段直接拒绝，
        # 而不是等训练进行到一半再失败。
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        # frozen Transition 保证实时环境推进到新帧时，已存经验不会随之改变。
        self._items.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        # 无放回抽样避免同一个 batch 内重复某条经验；传入独立 RNG
        # 可让抽样序列由随机种子复现。
        if batch_size > len(self._items):
            raise ValueError("batch size exceeds replay buffer size")
        return rng.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        # 只公开长度、追加和抽样，使训练代码不依赖经验池的内部容器实现。
        return len(self._items)


def epsilon_for_step(step: int) -> float:
    """返回线性退火后的探索概率。

    调度依据全局决策数而不是 episode 数，因为随着智能体学会生存，
    每局长度会发生显著变化。
    """
    # 同时限制上下界，使负输入或恢复训练后超过衰减窗口的步数
    # 仍能得到合法概率。
    fraction = min(1.0, max(0, step) / EPSILON_DECAY_STEPS)

    # 到达终点时直接返回常量，避免指标和测试出现
    # 0.050000000000000044 这类微小浮点误差。
    if fraction >= 1.0:
        return EPSILON_END
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


# 这是算法核心。普通 DQN 用同一组有噪声的估计既选动作又评估动作，
# 容易系统性高估；Double DQN 让在线网络负责“选谁”，目标网络负责
# “这个选择值多少”，再用得到的目标反向更新在线网络。
def optimize_model(
    online_network: DQN,
    target_network: DQN,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
) -> float:
    """执行一次带动作掩码的 Double DQN 更新，并返回标量损失。

    每条经验近似学习以下等式：
    ``Q(s,a) = r + gamma * Q_target(s', argmax Q_online(s', ·))``。
    将动作选择与动作估值分开，可降低普通 DQN 对含噪估计取最大值时
    产生的系统性正偏差。
    """
    # 经验池存储普通 Python 值以保持设备无关。只有开始更新时，抽出的 batch
    # 才转换为稠密张量，并直接送到选定的 CPU 或 CUDA 设备。
    states = torch.tensor(
        [item.state for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor(
        [item.action for item in transitions],
        dtype=torch.long,
        device=device,
    )
    rewards = torch.tensor(
        [item.reward for item in transitions],
        dtype=torch.float32,
        device=device,
    )

    # 下一状态及其动作掩码只用于等式的目标一侧；``done`` 决定未来价值项
    # 是否存在。
    next_states = torch.tensor(
        [item.next_state for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    dones = torch.tensor(
        [item.done for item in transitions],
        dtype=torch.bool,
        device=device,
    )
    masks = torch.tensor(
        [item.next_action_mask for item in transitions],
        dtype=torch.bool,
        device=device,
    )

    # 网络一次预测所有动作；``gather`` 只保留每条回放经验中实际执行动作的 Q 值。
    predicted = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # target 是本次更新的监督标签，不应沿它继续优化。no-grad 会阻止梯度
    # 通过任一网络形成第二条路径，并显著减少显存使用。
    with torch.no_grad():
        # argmax 前把非法动作设为负无穷；即使未约束网络意外给出乐观高值，
        # 这些动作也绝不可能被选中。
        online_next = online_network(next_states).masked_fill(~masks, -torch.inf)
        next_actions = online_next.argmax(dim=1)

        # 这里体现 Double DQN 的关键分工：online 选择 ``next_actions``，
        # 更新较慢的 target 网络只负责评估这些已选动作。
        target_next = target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # 终局经验包含最后奖励，但不存在未来回报；乘以 ``~dones`` 后，
        # 它们的自举贡献变为零。
        targets = rewards + GAMMA * target_next * (~dones)

    # Smooth L1 在零附近为二次函数，在大误差区域为线性函数。它对正常信号
    # 表现得类似 MSE，同时不会让少数离群值主导训练。
    loss = F.smooth_l1_loss(predicted, targets)

    # ``set_to_none`` 避免显式清零每个梯度缓冲区，让 PyTorch 只为
    # 反向传播实际到达的参数分配缓冲区。
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 在 AdamW 执行自适应更新前限制总梯度范数。它只是一道安全护栏，
    # 不能代替合理的奖励尺度设计。
    nn.utils.clip_grad_norm_(online_network.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()

    # 指标稍后写入 JSON，因此返回与设备无关的 float，而不是继续持有
    # Tensor 及其计算图。
    return float(loss.detach().cpu())


def _checkpoint_metadata(boss_hp: int, training_episodes: int, global_step: int) -> dict[str, object]:
    """在张量权重之外记录模型契约。

    单独的 state dictionary 只有各层张量，不包含观测含义、动作顺序或
    训练进度。以下字段可让不兼容或过期检查点明确失败，而不是表现出
    难以解释的异常行为。
    """
    # 基础身份字段用于判断文件属于哪种算法及哪套环境接口。
    return {
        "algorithm": "double-dqn",
        "actions": list(BossDodgeEnv.ACTIONS),
        # 状态与网络结构字段决定如何重建输入层、隐藏层和输出层。
        "state_encoding": STATE_ENCODING,
        "state_dimensions": STATE_DIMENSIONS,
        "hidden_dimensions": list(HIDDEN_DIMENSIONS),
        # 决策时间尺度和折扣率必须与采集这些经验时保持一致。
        "action_repeat": ACTION_REPEAT,
        "gamma": GAMMA,
        # 训练进度字段用于恢复探索率、随机种子序列和累计局数。
        "boss_hp": boss_hp,
        "training_episodes": training_episodes,
        "global_step": global_step,
    }


# 检查点不仅是参数文件，也是一份模型契约。动作顺序或状态编码改变后，
# 即使矩阵形状碰巧一致，其含义也已经不同，因此必须明确拒绝旧文件，
# 不能让一个“能加载但行为错误”的策略进入训练或可视化。
def validate_checkpoint(checkpoint: dict[str, object]) -> None:
    """拒绝当前环境无法正确解释的检查点。"""

    # 算法标识可防止把历史 JSON Q 表或其他价值学习检查点
    # 意外送入当前神经网络训练器。
    if checkpoint.get("algorithm") != "double-dqn":
        raise ValueError("checkpoint is not a Double DQN checkpoint")
    # Q 输出索引 0 之所以表示 "left"，完全依赖固定动作顺序。即使动作数量
    # 相同，只要排列发生改变，每一次决策的含义都会被破坏。
    if tuple(checkpoint.get("actions", ())) != BossDodgeEnv.ACTIONS:
        raise ValueError("checkpoint actions do not match the environment")
    # 如果特征含义或尺度已改变，仅张量维度一致仍不够，因此需要显式且
    # 带版本的状态编码标识。
    if checkpoint.get("state_encoding") != STATE_ENCODING:
        raise ValueError("checkpoint state encoding does not match the trainer")
    # 为有限兼容性，target 与优化器状态可以缺失；但无论推理还是恢复训练，
    # 都必须存在 online 网络权重。
    if "online_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain network weights")


def load_training_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    *,
    reset: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, object] | None:
    """除非要求全新训练，否则加载受信任的本地检查点。

    ``reset`` 的优先级高于文件是否存在，因此可以复用同一个命令行路径，
    同时明确丢弃旧学习状态。
    """
    # 返回 None 是给 train 函数的明确信号，表示两个网络都应从
    # 新的随机参数开始初始化。
    if reset:
        return None
    # 拒绝隐式开始新训练，可避免用户因路径拼写错误创建了新策略，
    # 却误以为自己成功恢复了长时间实验。
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}; use --reset to start a new DQN"
        )
    # 优化器状态和元数据不属于纯权重 state dictionary，因此此检查点
    # 只应从受信任的本地训练结果加载。
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    # 在把任何内部张量交给网络之前，同时验证外层序列化类型和语义契约。
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid checkpoint: {checkpoint_path}")
    validate_checkpoint(checkpoint)
    return checkpoint


def build_network_from_checkpoint(
    checkpoint: dict[str, object],
    device: str | torch.device = "cpu",
) -> DQN:
    """根据检查点元数据和权重重建推理网络。"""

    # 在索引元数据字段前先校验，使损坏文件尽可能给出针对检查点的明确错误。
    validate_checkpoint(checkpoint)
    target_device = torch.device(device)

    # 网络结构维度保存在检查点中，因此即使保存的是非默认隐藏层布局，
    # 也能无需修改代码直接进行可视化。
    network = DQN(
        state_dimensions=int(checkpoint["state_dimensions"]),
        action_count=len(checkpoint["actions"]),
        hidden_dimensions=tuple(int(value) for value in checkpoint["hidden_dimensions"]),
    ).to(target_device)
    # online 权重代表最新学到的策略；target 权重有意保持较旧，
    # 只用于稳定训练标签。
    network.load_state_dict(checkpoint["online_state_dict"])

    # eval 模式既明确表达推理意图，也为未来加入 dropout 或 batch norm
    # 等训练/推理行为不同的层做好准备。
    network.eval()
    return network


# 训练循环的节奏是：观察状态 -> 按 epsilon-greedy 选合法动作 ->
# 环境推进 -> 写入回放池 -> 随机抽一个 batch 更新在线网络。
# 每隔固定决策步，再把在线参数复制给目标网络，稳定学习目标。
def train(
    episodes: int = 3000,
    seed: int = 7,
    boss_hp: int = BossDodgeEnv.INITIAL_BOSS_HP,
    initial_checkpoint: dict[str, object] | None = None,
    *,
    device: str | torch.device | None = None,
    batch_size: int = BATCH_SIZE,
    replay_warmup: int = REPLAY_WARMUP,
) -> tuple[dict[str, object], dict[str, object]]:
    """训练 Double DQN 智能体，并返回指标和检查点。

    外层每次迭代对应一局 Boss 战；在一局内部，智能体反复选择动作、
    记录 transition，并在预热完成后随机抽取 replay batch 更新网络。
    """
    # 以下检查可防止循环和回放抽样接受非法配置；否则问题可能要等环境
    # 已运行一段时间后才暴露。
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if batch_size <= 0 or replay_warmup < 0:
        raise ValueError("batch_size must be positive and replay_warmup non-negative")
    # 自动选择会优先使用可用的 CUDA。由于模型较小，环境模拟和 Python
    # 经验回放操作仍可能占据总耗时的大部分。
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 环境探索、回放抽样和权重初始化使用不同随机生成器。为它们全部设种子
    # 可提高实验对比的可复现性，但 GPU 内核仍可能存在平台差异。
    random.seed(seed)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    # 即使手动游玩可以没有时间限制，训练也使用有限时域。DQN 输出数量
    # 直接取自环境动作契约，避免两处配置不一致。
    env = BossDodgeEnv(
        seed=seed,
        max_steps=MAX_EPISODE_STEPS,
        initial_boss_hp=boss_hp,
    )
    online = DQN(action_count=len(env.ACTIONS)).to(target_device)
    target = DQN(action_count=len(env.ACTIONS)).to(target_device)

    # AdamW 为每个参数自适应调整有效步长，并使用解耦权重衰减。
    # 只有 online 网络参数被直接优化。
    optimizer = torch.optim.AdamW(online.parameters(), lr=LEARNING_RATE)

    # global_step 在恢复训练前后持续驱动 epsilon 和目标网络同步；
    # episode 计数主要用于指标统计和确定性随机种子。
    previous_episodes = 0
    global_step = 0

    # 恢复训练会尽可能还原完整学习状态。较旧但兼容的检查点可能缺少 target
    # 或优化器状态，因此这些可选字段使用合理回退方案。
    if initial_checkpoint is not None:
        validate_checkpoint(initial_checkpoint)
        online.load_state_dict(initial_checkpoint["online_state_dict"])
        target.load_state_dict(
            initial_checkpoint.get(
                "target_state_dict",
                initial_checkpoint["online_state_dict"],
            )
        )
        if "optimizer_state_dict" in initial_checkpoint:
            optimizer.load_state_dict(initial_checkpoint["optimizer_state_dict"])
        previous_episodes = int(initial_checkpoint.get("training_episodes", 0))
        global_step = int(initial_checkpoint.get("global_step", 0))
    else:
        # 全新 target 必须与 online 初始参数完全一致；两个独立随机初始化
        # 会产生任意且误导性的首批学习目标。
        target.load_state_dict(online.state_dict())

    # target 输出永远不需要梯度；若未来加入有状态层，``eval`` 也能保证
    # 它按确定的推理方式工作。
    target.eval()

    # replay 有意只存在于当前训练会话，不做序列化，因为包含 50,000 条经验的
    # Python 对象会显著增大检查点。恢复训练后需重新预热一份回放历史。
    replay = ReplayBuffer()

    # episode 级列表用于绘制学习曲线；loss 按更新步记录，因为一局可能包含
    # 多次梯度更新，也可能在早期预热时一次都没有。
    rewards: list[float] = []
    boss_hits: list[int] = []
    losses: list[float] = []
    wins: list[int] = []
    timeouts: list[int] = []

    # 每次 reset 使用不同但可复现的种子；加入历史 episode 数可避免恢复训练后
    # 再次重复最初的随机局面。
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + previous_episodes + episode)

        # 经验池始终接收编码后的状态而非原始观测，确保训练与可视化
        # 使用完全相同的缩放方式。
        state = encode_state(observation)
        total_reward = 0.0
        hits = episode_timeouts = 0

        # 环境自行决定战斗何时自然结束或达到时间上限，因此决策循环
        # 只通过 ``done`` 退出。
        while True:
            available = available_action_indices(env)
            epsilon = epsilon_for_step(global_step)

            # epsilon-greedy 平衡探索与利用。两个分支使用同一合法动作集合，
            # 因而探索不会把步骤浪费在冷却动作或被墙阻挡的指令上。
            if rng.random() < epsilon:
                action = rng.choice(available)
            else:
                # 动作选择属于推理而非训练；禁用梯度可避免每个游戏步骤
                # 都保留一份计算图。
                with torch.no_grad():
                    state_tensor = torch.tensor(
                        state,
                        dtype=torch.float32,
                        device=target_device,
                    ).unsqueeze(0)
                    q_values = online(state_tensor).squeeze(0)
                action = select_greedy_action(q_values, rng, available)

            # 一个已选动作最多推进 ACTION_REPEAT 帧，并为 Bellman 等式
            # 产生一条决策级 transition。
            next_observation, reward, terminated, truncated, info = (
                step_with_action_repeat(env, action)
            )
            next_state = encode_state(next_observation)

            # 自然终止和训练时域截断都会停止传播未来价值；下一动作掩码
            # 在环境再次变化之前立即捕获。
            done = terminated or truncated
            replay.append(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    action_mask(env),
                )
            )

            # 推进实时状态，同时把便于人阅读的单局指标与回放学习表示
            # 分开累计。
            state = next_state
            total_reward += reward
            hits += int(info["boss_hit"])
            episode_timeouts += int(info["spike_escape_timeout"])
            # 一个 global step 表示一次 DQN 决策，而不是底层一帧；
            # 因此 epsilon 与 target 更新都和回放 transition 对齐。
            global_step += 1

            # 预热阈值还会与 batch 大小比较，确保调用者设置的 batch
            # 大于预热值时，抽样仍然合法。
            if len(replay) >= max(batch_size, replay_warmup):
                batch = replay.sample(batch_size, rng)
                loss = optimize_model(
                    online,
                    target,
                    optimizer,
                    batch,
                    target_device,
                )
                losses.append(loss)

            # 对这个小任务，硬同步简单且稳定；两次复制之间 target 参数固定，
            # online 网络则持续学习。
            if global_step % TARGET_UPDATE_INTERVAL == 0:
                target.load_state_dict(online.state_dict())

            # 单局指标只在结束时结算一次，数据来自最后一条环境 transition
            # 返回的终局 info 字典。
            if done:
                rewards.append(total_reward)
                boss_hits.append(hits)
                timeouts.append(episode_timeouts)
                wins.append(int(info["won"]))
                break

    # 检查点同时包含策略权重和继续优化所需状态；经验回放是上文说明的
    # 唯一有意不保存的例外。
    checkpoint = _checkpoint_metadata(
        boss_hp,
        previous_episodes + episodes,
        global_step,
    )
    checkpoint.update(
        online_state_dict=online.state_dict(),
        target_state_dict=target.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
    )
    # 指标只使用与 JSON 兼容的普通列表、浮点数、整数和字符串。
    # 保留每次更新的 loss，便于以后绘制详细的稳定性曲线。
    metrics: dict[str, object] = {
        # 以下列表按 episode 对齐，可共同绘制奖励、命中和胜负曲线。
        "algorithm": "double-dqn",
        "episode_rewards": rewards,
        "episode_boss_hits": boss_hits,
        "episode_wins": wins,
        "episode_spike_escape_timeouts": timeouts,
        # loss 按梯度更新对齐，数量通常多于 episode 数量。
        "training_losses": losses,
        # 最终探索率和设备信息用于解释不同实验之间的条件差异。
        "final_epsilon": epsilon_for_step(global_step),
        "boss_hp": boss_hp,
        "device": str(target_device),
    }
    # 持久化交给调用者负责，使测试可以完全在内存中训练，命令行也能
    # 自由选择其他恢复或输出路径。
    return metrics, checkpoint


# 评估阶段关闭随机探索，只观察网络真正学会的贪心策略。
# 它使用与训练不同的随机种子，避免把“记住训练局面”误认为泛化能力；
# 胜率、受伤和尖刺逃脱超时从不同角度描述策略质量。
def evaluate(
    checkpoint: dict[str, object],
    episodes: int = 100,
    seed: int = 1007,
    boss_hp: int | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """在未见随机种子上运行贪心策略，并汇总其行为。

    评估会关闭 epsilon 探索和梯度构建。将它与训练分开后，报告的胜率
    才能反映真正学到的策略，而不是采集经验时使用的随机探索动作。
    """
    # 只有 episode 数为正，平均指标才拥有有意义的分母。
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    # 默认按训练时记录的难度评估；显式覆盖 boss_hp 可测试同一网络
    # 是否能在不同生命值难度之间泛化。
    if boss_hp is None:
        boss_hp = int(checkpoint.get("boss_hp", BossDodgeEnv.INITIAL_BOSS_HP))
    # 从序列化形式重建网络会走与可视化器相同的加载路径，
    # 从而尽早发现检查点与网络结构不匹配。
    network = build_network_from_checkpoint(checkpoint, device)
    target_device = next(network.parameters()).device

    # 评估使用与训练相同的单局时域和动作重复次数；改变其中任何一项，
    # 都会使回报和成功率失去可比性。
    env = BossDodgeEnv(
        seed=seed,
        max_steps=MAX_EPISODE_STEPS,
        initial_boss_hp=boss_hp,
    )
    rng = random.Random(seed)
    wins = damage = spike_escape_timeouts = 0

    # 使用彼此不同但确定的种子，可在重复评估的同时，避免依赖训练环境
    # 最终停留的随机生成器状态。
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        while True:
            # 添加前导 batch 维度，因为 DQN 始终接收状态矩阵，
            # 即使当前一次只评估一个状态。
            state = torch.tensor(
                encode_state(observation),
                dtype=torch.float32,
                device=target_device,
            ).unsqueeze(0)

            # 评估不需要 target 网络：这里只询问最终 online 策略，
            # 当前哪个合法动作的估计价值最高。
            with torch.no_grad():
                values = network(state).squeeze(0)
            action = select_greedy_action(values, rng, available_action_indices(env))
            observation, _, terminated, truncated, info = step_with_action_repeat(
                env,
                action,
            )

            # 超时按聚合后的决策计数，因为合并后的 info 表示它是否在
            # 任一重复帧中发生。
            spike_escape_timeouts += int(info["spike_escape_timeout"])
            if terminated or truncated:
                # 受伤量和是否胜利是环境在终局暴露的单局汇总结果。
                wins += int(info["won"])
                damage += int(info["damage_taken"])
                break
    # 使用平均值后，不同 episode 数量的评估结果可以直接比较。
    return {
        "win_rate": wins / episodes,
        "average_damage_taken": damage / episodes,
        "average_spike_escape_timeouts": spike_escape_timeouts / episodes,
    }


def parse_bool(value: str) -> bool:
    """解析常见文本布尔形式，以保持程序调用兼容性。

    当前 CLI 把 ``--reset`` 作为开关，但保留此辅助函数，可兼容过去
    显式传入布尔文本的代码。
    """
    normalized = value.strip().lower()

    # 匹配用户输入前统一大小写并去除首尾空白。
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


# 命令行层只负责组织一次实验：读取旧检查点、调用训练、固定策略评估，
# 最后同时保存可恢复的 ``dqn.pt`` 和便于画曲线的 ``dqn.json``。
# 算法本身不依赖命令行，因此测试或其他程序可以直接调用 train/evaluate。
def main() -> None:
    """解析命令行选项，依次完成训练、评估和结果持久化。"""

    # 参数默认值定义正常完整实验；在投入较长 GPU 时间前，可用较小的
    # 训练局数和评估局数做冒烟测试。
    parser = argparse.ArgumentParser(description="Train a Double DQN boss-fight agent")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)

    # Boss HP 受环境支持范围约束。课程式训练可以先学习低 HP 的简单战斗，
    # 再恢复检查点挑战更高难度。
    parser.add_argument(
        "--boss-hp",
        type=int,
        choices=range(1, BossDodgeEnv.INITIAL_BOSS_HP + 1),
        default=BossDodgeEnv.INITIAL_BOSS_HP,
    )
    # ``--resume`` 同时选择输入与输出检查点路径；``--reset`` 会忽略已有内容，
    # 明确从随机权重开始。
    parser.add_argument(
        "--resume",
        type=Path,
        help="checkpoint to resume (default: checkpoints/dqn.pt)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="start a new network even if a checkpoint exists",
    )
    args = parser.parse_args()

    # 在这里给出清晰配置错误，而不是让第一次 CUDA 张量分配
    # 在训练循环深处才失败。
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    checkpoint_path = args.resume or DEFAULT_CHECKPOINT_PATH

    # 检查点缺失时必须显式 reset，防止长时间实验因错误路径而静默重启。
    try:
        initial = load_training_checkpoint(checkpoint_path, reset=args.reset)
    except FileNotFoundError as error:
        parser.error(str(error))
    # ``train`` 本身没有文件系统副作用；它返回可序列化结果，
    # 再由命令行编排层在成功完成后持久化。
    metrics, checkpoint = train(
        episodes=args.episodes,
        seed=args.seed,
        boss_hp=args.boss_hp,
        initial_checkpoint=initial,
        device=args.device,
    )
    # 评估使用偏移后的随机种子，不会简单重放训练局面序列。
    # 未强制指定设备时，CPU 是便于迁移的评估默认值。
    metrics["evaluation"] = evaluate(
        checkpoint,
        episodes=args.eval_episodes,
        seed=args.seed + 1000,
        boss_hp=args.boss_hp,
        device=args.device or "cpu",
    )
    # 只有训练和评估都成功后才创建输出目录，避免部分产物被误认为完整实验。
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch 序列化负责保留张量和优化器状态；指标只包含普通 Python 数据，
    # 因此使用可直接阅读的 JSON。
    torch.save(checkpoint, checkpoint_path)
    DEFAULT_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # 当策略已显著变化时，近期均值比全程均值更有参考价值；
    # 对短实验，这个切片会自然包含全部 episode。
    recent = metrics["episode_rewards"][-100:]
    print(
        f"episodes={args.episodes} mean_reward={sum(recent) / len(recent):.3f} "
        f"epsilon={metrics['final_epsilon']:.3f} checkpoint={checkpoint_path}"
    )
    print(json.dumps(metrics["evaluation"], indent=2))


# 导入模块只暴露可复用函数，不会自动开始漫长训练；
# 只有本文件作为选定入口执行时才调用 main。
if __name__ == "__main__":
    main()

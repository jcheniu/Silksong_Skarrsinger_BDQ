"""Train and evaluate an interpretable tabular Q-learning agent."""

from __future__ import annotations

# 导入本模块依赖的类型与运行时工具。
import argparse
import json
# 导入本模块依赖的类型与运行时工具。
from pathlib import Path
import random

# 导入本模块依赖的类型与运行时工具。
from .boss_env import BossDodgeEnv


ACTION_REPEAT = 2
# 配置 FRAME_GAMMA，统一约束后续计算使用的规则参数。
FRAME_GAMMA = 0.995
FRAME_LAMBDA = 0.99
# 配置 GAMMA，统一约束后续计算使用的规则参数。
GAMMA = FRAME_GAMMA ** ACTION_REPEAT
LAMBDA = FRAME_LAMBDA ** ACTION_REPEAT
# 配置 LEARNING_RATE，统一约束后续计算使用的规则参数。
LEARNING_RATE = 0.05
TRACE_THRESHOLD = 1e-5
# 配置 MAX_EPISODE_STEPS，统一约束后续计算使用的规则参数。
MAX_EPISODE_STEPS = 1200
STATE_ENCODING = "compact-v17-no-hp-repeat2"
# 配置 DEFAULT_CHECKPOINT_PATH，统一约束后续计算使用的规则参数。
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/q_table.json")
BOSS_OFFSET_BIN = 15
# 配置 BOSS_OFFSET_LIMIT，统一约束后续计算使用的规则参数。
BOSS_OFFSET_LIMIT = 8
ATTACK_URGENT_FRAMES = 8
# 配置 STATE_DIMENSIONS，统一约束后续计算使用的规则参数。
STATE_DIMENSIONS = 12

REPEATED_EVENT_INFO_KEYS = (
    "boss_hit",
    "boss_teleported",
    "spike_dodged",
    "entered_attack_range",
    "spike_escape_timeout",
)


# 定义 encode_state，集中处理这一阶段的输入与状态变化。
def encode_state(state: tuple[float | int, ...]) -> tuple[int, ...]:
    """Aggregate nearby observations into a compact, noise-tolerant state."""
    # 计算并保存 player_x、player_y、player_velocity_y、player_facing、player_attack_recovery_timer、player_dash_timer、player_dash_direction、player_dash_recovery_timer、boss_x、boss_y、_boss_velocity_x、attack_x、_attack_y、attack_phase、attack_timer、_player_hp、_boss_hp、invulnerable_timer，供后续逻辑直接复用。
    (
        player_x,
        player_y,
        player_velocity_y,
        player_facing,
        player_attack_recovery_timer,
        player_dash_timer,
        player_dash_direction,
        player_dash_recovery_timer,
        boss_x,
        boss_y,
        _boss_velocity_x,
        attack_x,
        _attack_y,
        attack_phase,
        attack_timer,
        _player_hp,
        _boss_hp,
        invulnerable_timer,
    ) = state

    # 计算并保存 player_center，供后续逻辑直接复用。
    player_center = player_x + BossDodgeEnv.PLAYER_WIDTH / 2
    boss_center = boss_x + BossDodgeEnv.BOSS_WIDTH / 2
    # 计算并保存 boss_offset，供后续逻辑直接复用。
    boss_offset = round((boss_center - player_center) / BOSS_OFFSET_BIN)
    boss_offset = max(-BOSS_OFFSET_LIMIT, min(BOSS_OFFSET_LIMIT, boss_offset))

    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if player_y == 0 and player_velocity_y == 0:
        vertical_state = 0
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    elif player_velocity_y > 0:
        vertical_state = 1
    # 处理前述条件未覆盖的其余情况。
    else:
        vertical_state = -1

    # 计算并保存 sword_overlaps_boss_y，供后续逻辑直接复用。
    sword_overlaps_boss_y = int(
        player_y < boss_y + BossDodgeEnv.BOSS_HEIGHT
        and player_y + BossDodgeEnv.SWORD_HEIGHT > boss_y
    )
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if player_facing > 0:
        sword_x = player_x + BossDodgeEnv.PLAYER_WIDTH
    # 处理前述条件未覆盖的其余情况。
    else:
        sword_x = player_x - BossDodgeEnv.SWORD_WIDTH
    # 计算并保存 boss_attackable，供后续逻辑直接复用。
    boss_attackable = int(
        sword_x < boss_x + BossDodgeEnv.BOSS_WIDTH
        and sword_x + BossDodgeEnv.SWORD_WIDTH > boss_x
        and bool(sword_overlaps_boss_y)
    )

    # 计算并保存 dash_state，供后续逻辑直接复用。
    dash_state = int(player_dash_direction) if player_dash_timer > 0 else 0
    if player_x <= 0:
        # 计算并保存 wall_state，供后续逻辑直接复用。
        wall_state = -1
    elif player_x >= BossDodgeEnv.ARENA_WIDTH - BossDodgeEnv.PLAYER_WIDTH:
        # 计算并保存 wall_state，供后续逻辑直接复用。
        wall_state = 1
    else:
        # 计算并保存 wall_state，供后续逻辑直接复用。
        wall_state = 0

    spike_escape_direction = 0
    # 计算并保存 spike_overlaps_player，供后续逻辑直接复用。
    spike_overlaps_player = (
        attack_x >= 0
        and attack_x < player_x + BossDodgeEnv.PLAYER_WIDTH
        and attack_x + BossDodgeEnv.BOSS_ATTACK_WIDTH > player_x
        and player_y < BossDodgeEnv.BOSS_ATTACK_HEIGHT
    )
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if spike_overlaps_player:
        attack_center = attack_x + BossDodgeEnv.BOSS_ATTACK_WIDTH / 2
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if player_center < attack_center:
            spike_escape_direction = -1
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        elif player_center > attack_center:
            spike_escape_direction = 1
        # 处理前述条件未覆盖的其余情况。
        else:
            spike_escape_direction = int(player_facing)

    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if attack_phase == BossDodgeEnv.ATTACK_WARNING:
        attack_urgency = 2 if attack_timer <= ATTACK_URGENT_FRAMES else 1
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    elif attack_phase == BossDodgeEnv.ATTACK_ACTIVE:
        attack_urgency = 3
    # 处理前述条件未覆盖的其余情况。
    else:
        attack_urgency = 0

    # 返回已经整理好的结果，供上层流程继续使用。
    return (
        vertical_state,
        sword_overlaps_boss_y,
        int(player_facing),
        int(player_attack_recovery_timer > 0),
        dash_state,
        int(player_dash_recovery_timer > 0),
        wall_state,
        boss_offset,
        boss_attackable,
        spike_escape_direction,
        attack_urgency,
        int(invulnerable_timer > 0),
    )


# 定义 available_action_indices，集中处理这一阶段的输入与状态变化。
def available_action_indices(env: BossDodgeEnv) -> tuple[int, ...]:
    """Return actions that can have an effect from the current frame."""
    # 计算并保存 available，供后续逻辑直接复用。
    available = set(range(len(env.ACTIONS)))
    if env.player_attack_recovery_timer > 0:
        # 调用 available.discard，推进当前处理步骤。
        available.discard(env.ACTIONS.index("attack"))
    if env.player_dash_recovery_timer > 0:
        # 调用 available.discard，推进当前处理步骤。
        available.discard(env.ACTIONS.index("dash"))
    if not env.is_grounded:
        # 调用 available.discard，推进当前处理步骤。
        available.discard(env.ACTIONS.index("jump"))
    if env.player_x <= 0:
        # 调用 available.discard，推进当前处理步骤。
        available.discard(env.ACTIONS.index("left"))
    if env.player_x >= env.ARENA_WIDTH - env.PLAYER_WIDTH:
        # 调用 available.discard，推进当前处理步骤。
        available.discard(env.ACTIONS.index("right"))
    return tuple(sorted(available))


# 定义 expected_epsilon_greedy_value，集中处理这一阶段的输入与状态变化。
def expected_epsilon_greedy_value(
    action_values: list[float],
    epsilon: float,
    available_actions: tuple[int, ...] | None = None,
) -> float:
    """Return the value expected under an epsilon-greedy policy."""
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if available_actions is None:
        available_actions = tuple(range(len(action_values)))
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if not available_actions:
        raise ValueError("available_actions must not be empty")
    # 计算并保存 available_values，供后续逻辑直接复用。
    available_values = [action_values[index] for index in available_actions]
    best_value = max(available_values)
    # 计算并保存 mean_value，供后续逻辑直接复用。
    mean_value = sum(available_values) / len(available_values)
    return (1.0 - epsilon) * best_value + epsilon * mean_value


# 定义 select_greedy_action，集中处理这一阶段的输入与状态变化。
def select_greedy_action(
    action_values: list[float],
    rng: random.Random,
    available_actions: tuple[int, ...] | None = None,
) -> int:
    """Choose uniformly among all actions tied for the highest Q value."""
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if not action_values:
        raise ValueError("action_values must not be empty")
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if available_actions is None:
        available_actions = tuple(range(len(action_values)))
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if not available_actions:
        raise ValueError("available_actions must not be empty")
    # 计算并保存 best_value，供后续逻辑直接复用。
    best_value = max(action_values[index] for index in available_actions)
    best_actions = [
        index
        for index in available_actions
        if action_values[index] == best_value
    ]
    # 返回已经整理好的结果，供上层流程继续使用。
    return rng.choice(best_actions)


def step_with_action_repeat(
    env: BossDodgeEnv,
    action: int,
) -> tuple[tuple[float | int, ...], float, bool, bool, dict[str, object]]:
    """Repeat one decision for two frames and combine its transition data."""
    # 计算并保存 total_reward，供后续逻辑直接复用。
    total_reward = 0.0
    repeated_events = {key: False for key in REPEATED_EVENT_INFO_KEYS}
    # 计算并保存 progress_penalty，供后续逻辑直接复用。
    progress_penalty = 0.0
    observation: tuple[float | int, ...] | None = None
    # 计算并保存 terminated，供后续逻辑直接复用。
    terminated = False
    truncated = False
    # 记录 info 字段，构成该对象对外提供的完整状态。
    info: dict[str, object] = {}
    frames_advanced = 0

    # 逐项处理当前序列，并累积这一轮所需的结果。
    for repeat_index in range(ACTION_REPEAT):
        observation, reward, terminated, truncated, frame_info = env.step(action)
        # 计算并保存 total_reward，供后续逻辑直接复用。
        total_reward += (FRAME_GAMMA ** repeat_index) * reward
        frames_advanced += 1
        # 计算并保存 info，供后续逻辑直接复用。
        info = dict(frame_info)
        for key in repeated_events:
            # 计算并保存 当前状态，供后续逻辑直接复用。
            repeated_events[key] = repeated_events[key] or bool(frame_info[key])
        progress_penalty += float(frame_info["progress_penalty"])
        # 根据当前条件选择对应分支，保持状态转换符合规则。
        if terminated or truncated:
            break

    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if observation is None:
        raise RuntimeError("action repeat did not advance the environment")
    # 调用 info.update，推进当前处理步骤。
    info.update(repeated_events)
    info["progress_penalty"] = progress_penalty
    # 计算并保存 当前状态，供后续逻辑直接复用。
    info["frames_advanced"] = frames_advanced
    return observation, total_reward, terminated, truncated, info


# 定义 deserialize_q_values，集中处理这一阶段的输入与状态变化。
def deserialize_q_values(
    q_data: dict[str, object],
    env: BossDodgeEnv,
) -> dict[tuple[int, ...], list[float]]:
    """Validate and restore a serialized Q table for continued training."""
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if tuple(q_data.get("actions", ())) != env.ACTIONS:
        raise ValueError("checkpoint actions do not match the current environment")
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if q_data.get("state_encoding") != STATE_ENCODING:
        raise ValueError("checkpoint state encoding does not match the trainer")
    # 计算并保存 serialized，供后续逻辑直接复用。
    serialized = q_data.get("q_values")
    if not isinstance(serialized, dict):
        # 输入或状态不满足约束时立即报告明确错误。
        raise ValueError("checkpoint does not contain a Q table")

    restored: dict[tuple[int, ...], list[float]] = {}
    # 逐项处理当前序列，并累积这一轮所需的结果。
    for key, row in serialized.items():
        state = tuple(int(part) for part in key.split("|"))
        # 计算并保存 values，供后续逻辑直接复用。
        values = [float(value) for value in row]
        if len(state) != STATE_DIMENSIONS or len(values) != len(env.ACTIONS):
            # 输入或状态不满足约束时立即报告明确错误。
            raise ValueError("checkpoint contains an invalid Q-table row")
        restored[state] = values
    # 返回已经整理好的结果，供上层流程继续使用。
    return restored


def parse_bool(value: str) -> bool:
    """Parse an explicit true/false command-line value."""
    # 计算并保存 normalized，供后续逻辑直接复用。
    normalized = value.strip().lower()
    if normalized == "true":
        # 返回已经整理好的结果，供上层流程继续使用。
        return True
    if normalized == "false":
        # 返回已经整理好的结果，供上层流程继续使用。
        return False
    raise argparse.ArgumentTypeError("expected true or false")


# 定义 load_training_checkpoint，集中处理这一阶段的输入与状态变化。
def load_training_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    *,
    reset: bool = False,
) -> dict[str, object] | None:
    """Load prior Q values unless the user explicitly requests a reset."""
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if reset:
        return None
    # 根据当前条件选择对应分支，保持状态转换符合规则。
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}; "
            "use --reset true to initialize a new Q table"
        )
    # 返回已经整理好的结果，供上层流程继续使用。
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def train(
    episodes: int = 3000,
    seed: int = 7,
    boss_hp: int = BossDodgeEnv.INITIAL_BOSS_HP,
    initial_q_data: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train with Expected SARSA(lambda) and replacing eligibility traces."""
    # 计算并保存 rng，供后续逻辑直接复用。
    rng = random.Random(seed)
    env = BossDodgeEnv(
        seed=seed,
        max_steps=MAX_EPISODE_STEPS,
        initial_boss_hp=boss_hp,
    )
    # 计算并保存 q，供后续逻辑直接复用。
    q = (
        {}
        if initial_q_data is None
        else deserialize_q_values(initial_q_data, env)
    )
    # 记录 rewards 字段，构成该对象对外提供的完整状态。
    rewards: list[float] = []
    boss_hits: list[int] = []
    # 记录 spike_escape_timeouts 字段，构成该对象对外提供的完整状态。
    spike_escape_timeouts: list[int] = []

    def values(state: tuple[int, ...]) -> list[float]:
        # 返回已经整理好的结果，供上层流程继续使用。
        return q.setdefault(state, [0.0] * len(env.ACTIONS))

    for episode in range(episodes):
        # 计算并保存 observation、_，供后续逻辑直接复用。
        observation, _ = env.reset(seed=seed + episode)
        state = encode_state(observation)
        # 计算并保存 epsilon，供后续逻辑直接复用。
        epsilon = max(0.05, 1.0 - episode / max(1, episodes * 0.8))
        eligibility: dict[tuple[tuple[int, ...], int], float] = {}
        # 计算并保存 total，供后续逻辑直接复用。
        total = 0.0
        hits = 0
        # 计算并保存 timeouts，供后续逻辑直接复用。
        timeouts = 0

        while True:
            # 计算并保存 available_actions，供后续逻辑直接复用。
            available_actions = available_action_indices(env)
            if rng.random() < epsilon:
                # 计算并保存 action，供后续逻辑直接复用。
                action = rng.choice(available_actions)
            else:
                # 计算并保存 action，供后续逻辑直接复用。
                action = select_greedy_action(
                    values(state),
                    rng,
                    available_actions,
                )

            # 计算并保存 next_observation、reward、terminated、truncated、info，供后续逻辑直接复用。
            next_observation, reward, terminated, truncated, info = (
                step_with_action_repeat(env, action)
            )
            # 计算并保存 next_state，供后续逻辑直接复用。
            next_state = encode_state(next_observation)
            total += reward
            # 计算并保存 hits，供后续逻辑直接复用。
            hits += int(info["boss_hit"])
            timeouts += int(info["spike_escape_timeout"])
            # 计算并保存 finished，供后续逻辑直接复用。
            finished = terminated or truncated

            expected_next = 0.0
            # 根据当前条件选择对应分支，保持状态转换符合规则。
            if not finished:
                expected_next = expected_epsilon_greedy_value(
                    values(next_state),
                    epsilon,
                    available_action_indices(env),
                )
            # 计算并保存 action_values，供后续逻辑直接复用。
            action_values = values(state)
            td_error = (
                reward
                + GAMMA * expected_next
                - action_values[action]
            )

            # 计算并保存 decay，供后续逻辑直接复用。
            decay = GAMMA * LAMBDA
            for state_action in list(eligibility):
                # 计算并保存 trace，供后续逻辑直接复用。
                trace = eligibility[state_action] * decay
                if trace < TRACE_THRESHOLD:
                    # 整理当前阶段的数据，随后进入下一步处理。
                    del eligibility[state_action]
                else:
                    # 计算并保存 当前状态，供后续逻辑直接复用。
                    eligibility[state_action] = trace
            eligibility[(state, action)] = 1.0

            # 逐项处理当前序列，并累积这一轮所需的结果。
            for (trace_state, trace_action), trace in eligibility.items():
                trace_values = values(trace_state)
                # 计算并保存 当前状态，供后续逻辑直接复用。
                trace_values[trace_action] += LEARNING_RATE * td_error * trace

            state = next_state
            # 根据当前条件选择对应分支，保持状态转换符合规则。
            if finished:
                break

        # 调用 rewards.append，推进当前处理步骤。
        rewards.append(total)
        boss_hits.append(hits)
        # 调用 spike_escape_timeouts.append，推进当前处理步骤。
        spike_escape_timeouts.append(timeouts)

    serial_q = {
        "|".join(map(str, state)): action_values
        for state, action_values in q.items()
    }
    # 计算并保存 previous_episodes，供后续逻辑直接复用。
    previous_episodes = (
        0
        if initial_q_data is None
        else int(initial_q_data.get("training_episodes", 0))
    )
    # 计算并保存 trained_levels，供后续逻辑直接复用。
    trained_levels = set(
        ()
        if initial_q_data is None
        else initial_q_data.get("trained_boss_hp_levels", ())
    )
    # 调用 trained_levels.add，推进当前处理步骤。
    trained_levels.add(boss_hp)
    return {
        "episode_rewards": rewards,
        "episode_boss_hits": boss_hits,
        "episode_spike_escape_timeouts": spike_escape_timeouts,
        "boss_hp": boss_hp,
    }, {
        "actions": list(env.ACTIONS),
        "state_encoding": STATE_ENCODING,
        "state_dimensions": STATE_DIMENSIONS,
        "boss_offset_bin": BOSS_OFFSET_BIN,
        "hp_in_state": False,
        "algorithm": "expected-sarsa-lambda",
        "action_repeat": ACTION_REPEAT,
        "frame_gamma": FRAME_GAMMA,
        "frame_lambda": FRAME_LAMBDA,
        "gamma": GAMMA,
        "lambda": LAMBDA,
        "trace_type": "replacing",
        "training_episodes": previous_episodes + episodes,
        "trained_boss_hp_levels": sorted(int(level) for level in trained_levels),
        "boss_hp": boss_hp,
        "q_values": serial_q,
    }


# 定义 evaluate，集中处理这一阶段的输入与状态变化。
def evaluate(
    q_data: dict[str, object],
    episodes: int = 100,
    seed: int = 1007,
    boss_hp: int | None = None,
) -> dict[str, float]:
    """Run a greedy policy without exploration and report simple metrics."""
    # 计算并保存 q_values，供后续逻辑直接复用。
    q_values = q_data["q_values"]
    if boss_hp is None:
        # 计算并保存 boss_hp，供后续逻辑直接复用。
        boss_hp = int(q_data.get("boss_hp", BossDodgeEnv.INITIAL_BOSS_HP))
    env = BossDodgeEnv(
        seed=seed,
        max_steps=MAX_EPISODE_STEPS,
        initial_boss_hp=boss_hp,
    )
    # 计算并保存 rng，供后续逻辑直接复用。
    rng = random.Random(seed)
    wins = 0
    # 计算并保存 damage，供后续逻辑直接复用。
    damage = 0
    spike_escape_timeouts = 0
    # 逐项处理当前序列，并累积这一轮所需的结果。
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        # 计算并保存 state，供后续逻辑直接复用。
        state = encode_state(observation)
        while True:
            # 计算并保存 key，供后续逻辑直接复用。
            key = "|".join(map(str, state))
            action_values = q_values.get(key, [0.0] * len(env.ACTIONS))
            # 计算并保存 action，供后续逻辑直接复用。
            action = select_greedy_action(
                action_values,
                rng,
                available_action_indices(env),
            )
            # 计算并保存 observation、_、terminated、truncated、info，供后续逻辑直接复用。
            observation, _, terminated, truncated, info = step_with_action_repeat(
                env,
                action,
            )
            # 计算并保存 spike_escape_timeouts，供后续逻辑直接复用。
            spike_escape_timeouts += int(info["spike_escape_timeout"])
            state = encode_state(observation)
            # 根据当前条件选择对应分支，保持状态转换符合规则。
            if terminated or truncated:
                wins += int(info["won"])
                # 计算并保存 damage，供后续逻辑直接复用。
                damage += int(info["damage_taken"])
                break
    # 返回已经整理好的结果，供上层流程继续使用。
    return {
        "win_rate": wins / episodes,
        "average_damage_taken": damage / episodes,
        "average_spike_escape_timeouts": spike_escape_timeouts / episodes,
    }


# 定义 main，集中处理这一阶段的输入与状态变化。
def main() -> None:
    parser = argparse.ArgumentParser()
    # 调用 parser.add_argument，推进当前处理步骤。
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument(
        "--boss-hp",
        type=int,
        choices=range(1, BossDodgeEnv.INITIAL_BOSS_HP + 1),
        default=BossDodgeEnv.INITIAL_BOSS_HP,
    )
    # 调用 parser.add_argument，推进当前处理步骤。
    parser.add_argument(
        "--resume",
        type=Path,
        help="load a non-default checkpoint (default: checkpoints/q_table.json)",
    )
    # 调用 parser.add_argument，推进当前处理步骤。
    parser.add_argument(
        "--reset",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help="initialize an empty Q table instead of loading the checkpoint",
    )
    # 计算并保存 args，供后续逻辑直接复用。
    args = parser.parse_args()
    checkpoint_path = args.resume or DEFAULT_CHECKPOINT_PATH
    # 执行可能失败的操作，并在后续分支中处理异常。
    try:
        initial_q_data = load_training_checkpoint(
            checkpoint_path,
            reset=args.reset,
        )
    # 收束异常路径，保证流程能够得到确定结果。
    except FileNotFoundError as error:
        parser.error(str(error))
    # 计算并保存 metrics、q_data，供后续逻辑直接复用。
    metrics, q_data = train(
        episodes=args.episodes,
        boss_hp=args.boss_hp,
        initial_q_data=initial_q_data,
    )
    # 计算并保存 当前状态，供后续逻辑直接复用。
    metrics["evaluation"] = evaluate(q_data, boss_hp=args.boss_hp)
    Path("runs").mkdir(exist_ok=True)
    # 调用 相关逻辑.mkdir，推进当前处理步骤。
    Path("checkpoints").mkdir(exist_ok=True)
    Path("runs/q_learning.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    # 调用 DEFAULT_CHECKPOINT_PATH.write_text，推进当前处理步骤。
    DEFAULT_CHECKPOINT_PATH.write_text(
        json.dumps(q_data, indent=2),
        encoding="utf-8",
    )
    # 计算并保存 recent，供后续逻辑直接复用。
    recent = metrics["episode_rewards"][-100:]
    recent_hits = metrics["episode_boss_hits"][-100:]
    # 调用 print，推进当前处理步骤。
    print(
        f"episodes={args.episodes} recent_mean_reward={sum(recent) / len(recent):.3f} "
        f"recent_boss_hits={sum(recent_hits)}"
    )
    # 调用 print，推进当前处理步骤。
    print(json.dumps(metrics["evaluation"], indent=2))


if __name__ == "__main__":
    # 调用 main，推进当前处理步骤。
    main()

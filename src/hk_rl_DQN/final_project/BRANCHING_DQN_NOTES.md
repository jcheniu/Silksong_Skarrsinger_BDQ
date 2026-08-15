# Branching DQN Design Notes

记录日期：2026-08-13

## 决策

最终真实环境采用 Multi-Discrete / Branching DQN，而不是当前一次只选择
一个整数动作的普通 DQN。策略网络每个控制周期输出一个动作 Tensor/list，
每个动作分支独立选择，因此移动、跳跃、攻击可以同时发生。

当前 `train_dqn.py` 和 `boss_env.py` 仍是单动作实现。这份文档描述下一阶段
改造目标，不表示训练链路已经完成改造。

## 动作分支

建议动作空间：

```python
MultiDiscrete([3, 7, 7])
```

分支顺序必须固定，并保存在检查点元数据中：

```text
jump_z:  [released, press_z, hold_z]
movement:[neutral, hold_left, hold_right, dash, left_dash, right_dash, pulse_s]
combat:  [neutral, tap_x, hold_x, press_shift, hold_v, up_x, down_x]
```

Movement exploration uses weights `32/32/12/8/8/8` for left, right, dash,
left-dash, right-dash, and S. Exploratory left/right selections persist for
2-3 total ticks. S is a one-tick pulse followed by an approximately 900 ms
active/recovery commitment: jump, movement, and combat are neutral during the
lock. Replay stores this coordinated action, while JSONL also preserves the
policy's attempted vector. S damage credits the movement head and an S miss has
no offensive-miss penalty.

例如：

```python
tensor([2, 5, 1])
```

表示同一个控制周期内同时执行 `right + jump + X`。各分支默认互不干扰；
只有同一分支内部互斥，例如 `left` 和 `right` 不能同时选择。

## 按键持续状态

网络输出应描述当前控制周期希望保持的按键状态，而不是高层技能名称：

- `X` 短按并释放是普通攻击。
- 连续多个周期保持 `X=held` 会累计蓄力。
- 蓄力达到 1350 ms（约 14 个 100 ms 控制周期）才算完成。
- `X=released`、受击或游戏 FSM 中断时重置蓄力。
- `C` 短按是冲刺，持续按住是快跑。
- `S` 对应移动头中的 `KeySupDash`/harpoon dash；它不消耗灵丝，合法性读取
  `HeroController.CanHarpoonDash()`。回血键 `A` 不进入动作空间，也不允许执行器发送。
- `LeftShift` 消耗灵丝，执行器结合当前灵丝、技能消耗、禁用状态以及
  游戏控制/冷却判定做合法性检查。

状态观测不再逐键展开。执行器把上一帧实际动作压缩为跳跃状态、移动方向、
移动方式、战斗类别、X 蓄力进度和 S 主动/后摇阶段共 6 维；动作可用性继续由
每个分支的 mask 负责。

不设置 `wall_jump` 分支。墙跳由 Agent 组合 `jump_z + left/right` 自行学习。

## 网络结构

使用共享状态编码器和多个 Q-value 输出头：

```python
features = shared_network(state)

jump_q = jump_head(features)          # 3
movement_q = movement_head(features)  # 7
combat_q = combat_head(features)      # 7
```

每个输出头分别执行 `argmax`，然后拼成 `[batch, branch_count]` 动作 Tensor。
不能把所有头拼平后只做一次全局 `argmax`，否则仍然只会产生一个动作。

优先参考 Branching Dueling Q-Network（BDQ）的共享 value stream 和每分支
advantage stream。共享环境奖励可以用于所有分支，但训练时应对所选分支 Q 值
求和或求平均，并保持 online/target 网络的 Double DQN 选择与估值分工一致。

## 需要修改的训练链路

1. `Transition.action: int` 改为固定长度动作向量。
2. Replay Buffer 保存形状为 `[branch_count]` 的动作 Tensor。
3. `BossDodgeEnv.step(int)` 改为接收并验证 Multi-Discrete 向量。
4. 环境在同一 tick 应用全部分支，而不是按动作名互斥处理。
5. Double DQN 对每个分支分别选择下一动作，并从 target 网络 gather 对应值。
6. epsilon-greedy 每个控制帧只判定一次；探索时按分支激活率稀疏采样合法组合，避免所有二元技能以 50% 概率同时按下。
7. 动作掩码改成每分支掩码，结合灵丝、生命值、FSM 和冷却状态。
8. 检查点保存分支名称、每分支大小、控制周期和动作协议版本。
9. 真实执行器读取动作 Tensor/list，并把所有 held 分支转换为同时按键。
10. JSONL 同时记录策略尝试向量和经过平滑、掩码及片段续接后的实际执行向量。

## 实施顺序

先在模拟环境中完成动作向量接口、BDQ 网络、回放训练和单元测试；确认能够
学习组合动作后，再编写 Tensor/list 到真实键盘输入的独立执行脚本。不要直接
覆盖当前可用的单动作训练器，保留它作为回归基线，并使用新的检查点版本号。

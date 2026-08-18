# Silksong Skarrsinger RL

[English](README_EN.md) | 简体中文

本项目使用强化学习控制《空洞骑士：丝之歌》中的 Skarrsinger Karmelita Boss 战。当前主线已从教学模拟器迁移到 Windows 实机闭环：BepInEx 插件输出只读遥测，Python 训练器编码状态、选择联合动作并发送键盘输入。旧版 Q-learning 和模拟器 DQN 已归档到 `history/`。

## 当前进度

| 模块 | 状态 | 当前实现 |
|---|---|---|
| 教学环境 | 已完成并归档 | `src/hk_rl/` 保留最小 Boss 闪避环境；两个历史快照位于 `history/` |
| Karmelita 练习循环 | 已实现 | 插件加载存档 1、进入 `Memory_Ant_Queen`、跳过前置敌人，并在死亡后重载；`F8` 可手动重载 |
| 实机遥测 | 已实现 | 每 50 ms 输出玩家/Boss 运动、资源、控制可用性、FSM、攻击生命周期、招架和 encounter ID |
| 状态编码 | 已实现 | 24 维归一化观察：10 维运动、1 维灵丝、7 维 Boss 语义、6 维执行器状态 |
| 动作执行 | 已实现 | 三字段语义协议 `[3,6,6]`，经规则筛选后提供 53 个联合动作，并支持运行时掩码和持续动作锁 |
| DQN | 已实现 | 24 -> 96 -> 96 的 Dueling Double DQN，单个 53 维联合动作优势头，共 16,950 个可训练参数 |
| 奖励与归因 | 已实现 | Boss 攻击窗口、受伤、命中、招架、攻击落空、灵丝消耗、边界和碰撞风险等即时/延迟奖励 |
| 训练恢复 | 已实现 | Replay Buffer、优化器、步数和回合数随检查点保存；协议不兼容时拒绝静默覆盖 |
| 实机训练结果 | 尚未形成可复现实验结论 | 仓库未提交足以报告胜率、平均受伤或收敛趋势的正式训练记录 |

玩家血量和原始 Boss HP 不进入 DQN 观察。玩家掉血和已确认的 Boss 伤害只用于奖励计算，避免策略直接读取目标血量。

## 当前架构

```text
BepInEx plugin -> telemetry.jsonl -> 24-value state encoder
                                      |
                                      v
                              Dueling Double DQN
                                      |
                                      v
                         53 curated joint actions
                                      |
                                      v
                         keyboard executor -> game
```

动作字段如下：

```text
jump_z:   [release, press_z, hold_z]
movement: [neutral, left, right, left_dash, right_dash, harpoon_s]
combat:   [neutral, tap_x, hold_x, shift, up_x, down_x]
```

53 个动作不是三个独立动作头的笛卡尔积训练结果，而是经过兼容性规则筛选的完整联合动作。`S` harpoon dash 是原子动作；`A` 回血、`D` 梦钉、`V` 嘲讽和通用朝向冲刺不在当前策略动作空间中。

关键训练参数：Replay 容量 50,000，预热 2,000 条 transition，batch size 128，折扣率 0.995，目标网络每 1,000 个训练 transition 更新。Epsilon 从 0.60 按 transition 衰减至 0.05；每 10 个训练回合插入一次不写 Replay、不更新网络的贪心评估回合。

更完整的动作、奖励和检查点协议见 [`src/hk_rl_DQN/README_EN.md`](src/hk_rl_DQN/README_EN.md) 与 [`src/hk_rl_DQN/final_project/BRANCHING_DQN_NOTES.md`](src/hk_rl_DQN/final_project/BRANCHING_DQN_NOTES.md)。

## 安装

需要 Python 3.11+。实机键盘控制和插件运行面向 Windows。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -r src\hk_rl_DQN\requirements.txt
```

构建插件前，将 `src/hk_rl_DQN/external/karmelita_practice/SilksongPath.props.example` 复制为同目录下的 `SilksongPath.props`，并填写本机游戏目录。详细安装、禁用和恢复步骤见 [`src/hk_rl_DQN/external/karmelita_practice/README.md`](src/hk_rl_DQN/external/karmelita_practice/README.md) 和 [`RECOVERY.md`](src/hk_rl_DQN/external/karmelita_practice/RECOVERY.md)。

## 运行

不发送键盘输入的管线检查：

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 1
```

游戏和插件准备好后启用实机控制：

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 10 --launch --execute-actions
```

默认情况下，存在兼容检查点就续训，不存在就新建。只有显式传入 `--reset` 才会清空模型、优化器、Replay 和训练计数。干跑模式会记录建议动作，但不会把未实际执行的动作加入 Replay 或用于梯度更新。

## 验证状态

与当前 53 动作实现一致的回归测试可这样运行：

```powershell
python -m pytest -q src\hk_rl_DQN\test_curated_actions.py tests\hk_rl_DQN\test_real_state.py
```

截至 2026-08-18，上述测试为 **26 passed**。

完整 `python -m pytest -q` 当前为 **89 passed, 42 failed**。失败主要来自 `tests/hk_rl_DQN/` 中仍针对早期 `[3,7,7]`、`V` 嘲讽和旧奖励常量的回归用例；这些用例尚未随当前 `[3,6,6]`、53 动作协议更新。因此，仓库当前实现可以被针对性测试覆盖，但全量测试基线尚未恢复为绿色。

## 目录

```text
src/hk_rl_DQN/                 当前实机训练主线
  external/karmelita_practice/ BepInEx 插件与遥测
  final_project/               动作目录、执行器和设计说明
  tools/                       冷启动与键位验收工具
src/hk_rl/                     教学用模拟环境
tests/                         自动化测试（部分仍对应旧协议）
docs/                          学习资料和设计文档
history/                       冻结的旧版 Q-learning/DQN 实现
```

运行产生的 `runs/`、`checkpoints/`、本机插件配置和游戏路径均被 Git 忽略。

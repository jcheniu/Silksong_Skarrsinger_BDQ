# Hollow Knight RL

本项目研究通过强化学习控制《空洞骑士：丝之歌》中的 Boss 战。当前主线已经从模拟器迁移到真实游戏遥测和键盘控制，旧模拟器仍保存在 `history/` 中作为参考。

## 当前主线

`src/hk_rl_DQN/` 是当前实机训练实现：

- `real_state.py`：24 维状态编码：10 维运动、1 维灵丝、7 维 Boss 语义和 6 维压缩动作状态。玩家血量不进入观察。
- `real_reward.py`：实时奖励计算，不使用 Boss HP。
- `final_project/action_executor.py`：3 个并行动作头、资源/FSM 合法性掩码和按键持续。
- `real_dqn.py`：Branching Dueling Double DQN、Replay、检查点和遥测训练循环。
- `external/karmelita_practice/`：BepInEx 遥测与 Karmelita 练习插件源码。
- `tools/`：冷启动验收和键位检查工具。

动作协议为 `[3,7,7]`：跳跃、移动和战斗。`S` harpoon dash 位于移动头，作为高位移冲刺；攻击、Shift 快速施法和嘲讽位于战斗头。回血键 `A` 和梦钉键 `D` 均不会发送。

移动探索以左右移动为主，普通/定向冲刺和 S 占较小但非零概率。探索产生的左右移动持续 2-3 个控制 tick；S 只按一个 tick，随后在主动段和后摇段内锁定其他动作。命中仍奖励 movement 头，未命中不惩罚。

## 安装与测试

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:PYTHONPATH='src'
python -m pytest -q
```

构建 BepInEx 插件前，复制 `src/hk_rl_DQN/external/karmelita_practice/SilksongPath.props.example` 为 `SilksongPath.props`，并填写本机游戏目录。

## 实机训练

不传 `--reset` 时，有检查点就续训，没有检查点就新建；损坏或协议不匹配的已有检查点会直接报错。只有明确传入 `--reset` 才会清零训练状态。

检查点同时保存模型、优化器和 Replay Buffer。中断后再次运行会恢复历史 transition，不会重新经历空 Replay 的 1000 step 预热；`--reset` 会同时清空 Replay。

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 3 --launch --execute-actions
```

先用少量回合验证游戏循环、奖励和动作日志，再扩大训练规模。干跑模式不会把未实际执行的虚拟动作写入 Replay。

## 目录

详见 [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md)。旧实现位于 `history/simulator_dqn_v1/` 和 `history/phase1_q_table_v1/`，不作为当前运行包导入。

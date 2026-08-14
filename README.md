# Hollow Knight RL

本项目研究通过强化学习控制《空洞骑士：丝之歌》中的 Boss 战。当前主线已经从模拟器迁移到真实游戏遥测和键盘控制，旧模拟器仍保存在 `history/` 中作为参考。

## 当前主线

`src/hk_rl_DQN/` 是当前实机训练实现：

- `real_state.py`：42 维状态编码，包含位置、速度、Boss FSM、反应、灵丝和按键持续状态。
- `real_reward.py`：实时奖励计算，不使用 Boss HP。
- `final_project/action_executor.py`：8 分支同时按键、资源/FSM 合法性掩码和按键持续。
- `real_dqn.py`：Branching Dueling Double DQN、Replay、检查点和遥测训练循环。
- `external/karmelita_practice/`：BepInEx 遥测与 Karmelita 练习插件源码。
- `tools/`：冷启动验收和键位检查工具。

动作协议为 `[3,3,3,2,2,2,2,2]`：水平、跳跃/二段跳、冲刺/跑步、攻击、S 技能、Shift 快速施法、禁用槽、嘲讽。回血键 `A` 和梦钉键 `D` 均不会发送。

## 安装与测试

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:PYTHONPATH='src'
python -m pytest -q
```

构建 BepInEx 插件前，复制 `src/hk_rl_DQN/external/karmelita_practice/SilksongPath.props.example` 为 `SilksongPath.props`，并填写本机游戏目录。该文件被 Git 忽略，不会上传本机路径。

## 实机训练

不传 `--reset` 时，有检查点就续训，没有检查点就新建；损坏或协议不匹配的已有检查点会直接报错。只有明确传入 `--reset` 才会清零训练状态。

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 3 --launch --execute-actions
```

先用少量回合验证游戏循环、奖励和动作日志，再扩大训练规模。干跑模式不会把未实际执行的虚拟动作写入 Replay。

## 目录

详见 [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md)。旧实现位于 `history/simulator_dqn_v1/` 和 `history/phase1_q_table_v1/`，不作为当前运行包导入。

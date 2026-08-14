# 第一阶段封存：Q-table 基线 v1

封存日期：2026-07-28

这是 Hollow Knight RL 项目第一阶段的独立快照。它保存了可运行的二维 Boss
环境、表格型智能体、可视化器、回归测试、训练产物和评估记录。后续 DQN
开发不应直接修改本目录；需要回溯或对照时，从这里读取或另行复制。

## 阶段结论

- 实际算法：Expected SARSA(lambda)，使用 replacing eligibility traces。
- 策略载体：Q table，不包含神经网络。
- 状态编码：`compact-v17-no-hp-repeat2`，12 个离散维度。
- 动作：`left`、`right`、`dash`、`attack`、`jump`、`wait`。
- 训练累计：50,000 episodes，覆盖 Boss HP 1 至 5。
- Q-table：10,130 个已访问状态。
- 保存的 100 局评估：胜率 81%，平均受伤 1.43，平均尖刺逃脱超时 0.98。
- 2026-07-28 固定种子复评：胜率 83%，平均受伤 1.43，平均尖刺逃脱超时 0.97。
- 回归测试：57 项全部通过。

“Q-table 阶段”是项目目标层面的名称；代码中的更新规则已经从基础 Q-learning
升级为 Expected SARSA(lambda)。保留这个区别，方便以后与 DQN 的 TD target、
replay buffer 和 target network 逐项比较。

## 内容

```text
hk_rl/                  第一阶段源码及其回归测试
tests/                  项目入口层的额外回归测试
checkpoints/q_table.json 训练后的 Q table
runs/q_learning.json     最近一次训练曲线与评估摘要
manifest.json            算法、成果和文件校验信息
verify_snapshot.py       快照完整性与可选复评工具
```

## 验证

在本目录执行：

```powershell
python verify_snapshot.py
python -m pytest -q -p no:cacheprovider
python verify_snapshot.py --evaluate
```

前两条只读取封存内容。`--evaluate` 使用保存的 Q table 重新运行 100 局固定种子
评估，也不会改写文件。Turtle 策略回放可用：

```powershell
python -m hk_rl.visualize --checkpoint checkpoints/q_table.json
```

不要在本目录执行训练入口，因为训练脚本会覆盖 `runs/` 和 `checkpoints/` 中的
阶段产物。新的 DQN 实现应放在封存目录之外，并复用或显式适配环境接口。

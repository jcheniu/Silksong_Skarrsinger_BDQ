# Project Layout

```text
src/hk_rl_DQN/                 当前真实游戏 Branching DQN
  external/karmelita_practice/ BepInEx C# 遥测插件源码
  final_project/               动作协议、执行器和设计说明
  tools/                       冷启动验收与本机键位工具
src/hk_rl/                     教学用模拟环境基线
tests/                         当前自动化测试
docs/                          学习资料、设计文档和验收说明
history/simulator_dqn_v1/      旧单动作模拟器 DQN
history/phase1_q_table_v1/     更早的 Q-learning 模拟器
history/manual_runs_*/         本机动作实验记录，不进入版本库
```

运行生成物默认写入根目录的 `runs/` 和 `checkpoints/`。这些目录被 Git 忽略，避免把模型、日志和本机实验状态混入源码。旧实验记录仍保存在 `history/`，但当前训练不依赖它们。

`SilksongPath.props`、游戏存档、BepInEx 安装缓存和 C# `bin/obj` 都是本机文件，不应提交。构建配置模板是 `SilksongPath.props.example`。

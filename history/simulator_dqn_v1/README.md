# Archived Simulator DQN v1

This directory is a frozen copy of the earlier rough, single-action simulator
DQN. It is retained as a regression reference and is not imported by the live
`src/hk_rl_DQN` package.

- `src/hk_rl_DQN`: simulator, single-action Double DQN, action protocol, and
  visualization code.
- `tests`: the matching simulator tests.
- `artifacts/checkpoints/dqn.pt`: the old trained checkpoint.
- `artifacts/runs/dqn.json`: the old training metrics.
- `OLD_README.md`: original run instructions, preserved for context.

The active real-game implementation uses an incompatible 42-value state and
eight-branch action protocol. Do not load this archive's checkpoint with
`hk_rl_DQN.real_dqn`.

Historical test status: 47 tests pass. One archived visualization test still
patches `hk_rl.visualize` although this package is named `hk_rl_DQN`; that known
failure is preserved rather than rewriting the frozen baseline.

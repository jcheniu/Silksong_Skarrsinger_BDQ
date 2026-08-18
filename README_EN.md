# Silksong Skarrsinger RL

English | [简体中文](README.md)

This project applies reinforcement learning to the Skarrsinger Karmelita boss encounter in *Hollow Knight: Silksong*. The active line has moved from teaching simulators to a Windows live-game loop: a BepInEx plugin emits read-only telemetry, and a Python trainer encodes state, selects joint actions, and sends keyboard input. Earlier Q-learning and simulator DQN versions are archived under `history/`.

## Current Progress

| Component | Status | Current implementation |
|---|---|---|
| Teaching environment | Complete and archived | `src/hk_rl/` retains the minimal boss-dodging environment; two frozen snapshots are under `history/` |
| Karmelita practice loop | Implemented | Loads save slot 1, enters `Memory_Ant_Queen`, skips the pre-boss wave, reloads after death, and supports manual reload with `F8` |
| Live telemetry | Implemented | Emits player/Boss motion, resources, control availability, FSM state, attack lifecycle, parries, and encounter IDs every 50 ms |
| State encoding | Implemented | 24 normalized values: 10 motion, 1 silk, 7 Boss semantics, and 6 executor-control values |
| Action execution | Implemented | Three-field `[3,6,6]` semantic protocol, reduced by compatibility rules to 53 joint actions with runtime masks and temporal locks |
| DQN | Implemented | `24 -> 96 -> 96` Dueling Double DQN with one 53-value joint-action advantage head and 16,950 trainable parameters |
| Reward and credit | Implemented | Immediate and delayed credit for Boss attack windows, damage, hits, parries, misses, silk use, arena boundaries, and collision risk |
| Training recovery | Implemented | Checkpoints persist replay, optimizer state, steps, and episodes; incompatible protocols are rejected instead of silently overwritten |
| Live training results | No reproducible conclusion yet | The repository does not contain a formal run with enough data to report win rate, average damage, or convergence |

Player health and raw Boss HP are excluded from the DQN observation. Player health loss and confirmed Boss damage are used only by the reward system, so the policy cannot directly read target health.

## Architecture

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

The action fields are:

```text
jump_z:   [release, press_z, hold_z]
movement: [neutral, left, right, left_dash, right_dash, harpoon_s]
combat:   [neutral, tap_x, hold_x, shift, up_x, down_x]
```

The 53 outputs are reviewed complete joint actions, not independently trained branches over the Cartesian product. Harpoon `S` is atomic. Healing `A`, Dream Nail `D`, taunt `V`, and a generic facing-based dash are absent from the current policy action space.

Key training settings are a 50,000-transition replay buffer, 2,000-transition warmup, batch size 128, discount 0.995, and target updates every 1,000 training transitions. Epsilon decays by training transition from 0.60 to 0.05. One greedy evaluation encounter runs after every 10 training episodes without changing replay, gradients, global step, or the training episode count.

See [`src/hk_rl_DQN/README_EN.md`](src/hk_rl_DQN/README_EN.md) and [`src/hk_rl_DQN/final_project/BRANCHING_DQN_NOTES.md`](src/hk_rl_DQN/final_project/BRANCHING_DQN_NOTES.md) for the detailed action, reward, and checkpoint protocols.

## Installation

Python 3.11+ is required. Live keyboard control and the plugin target Windows.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -r src\hk_rl_DQN\requirements.txt
```

Before building the plugin, copy `src/hk_rl_DQN/external/karmelita_practice/SilksongPath.props.example` to `SilksongPath.props` in the same directory and set the local game path. See the plugin [`README.md`](src/hk_rl_DQN/external/karmelita_practice/README.md) and [`RECOVERY.md`](src/hk_rl_DQN/external/karmelita_practice/RECOVERY.md) for installation, disable, and recovery procedures.

## Running

Run the pipeline without keyboard input:

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 1
```

Enable live control only after the game and plugin are ready:

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 10 --launch --execute-actions
```

By default, a compatible checkpoint is resumed and a missing checkpoint starts a new run. Only `--reset` clears the model, optimizer, replay, and training counters. Dry-run mode records proposed actions but does not add unexecuted actions to replay or use them for gradient updates.

## Validation Status

Run the regression tests aligned with the current 53-action implementation:

```powershell
python -m pytest -q src\hk_rl_DQN\test_curated_actions.py tests\hk_rl_DQN\test_real_state.py
```

As of 2026-08-18, this selection reports **26 passed**.

The complete `python -m pytest -q` currently reports **89 passed, 42 failed**. Most failures come from cases in `tests/hk_rl_DQN/` that still target the earlier `[3,7,7]` protocol, `V` taunt, and older reward constants. They have not yet been migrated to the current `[3,6,6]`, 53-action protocol. The active implementation therefore has focused regression coverage, but the repository-wide test baseline is not yet green.

## Layout

```text
src/hk_rl_DQN/                 active live-game training pipeline
  external/karmelita_practice/ BepInEx plugin and telemetry
  final_project/               action catalog, executor, and design notes
  tools/                       cold-start and key-binding checks
src/hk_rl/                     teaching simulator
tests/                         automated tests; some target the old protocol
docs/                          learning material and design documents
history/                       frozen earlier Q-learning/DQN implementations
```

Generated `runs/`, `checkpoints/`, local plugin configuration, and machine-specific game paths are ignored by Git.

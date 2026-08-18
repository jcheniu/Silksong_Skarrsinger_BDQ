# Silksong Karmelita Joint-Action DQN

This package trains a live-game reinforcement-learning agent for the Karmelita
encounter. It consumes telemetry emitted by the bundled BepInEx plugin and
controls the game through a coordinated keyboard-action executor.

The policy is a Dueling Double DQN over complete joint actions. It does not use
independently trained action branches.

## Architecture

The observation contains 24 normalized values:

```text
motion:   player x/y, player vx/vy, relative Boss x/y,
          relative Boss-player vx/vy, grounded, player facing       (10)
resource: normalized silk                                             (1)
Boss:     behavior progress, attack category, aerial, collision risk,
          vertical intent, hit pattern, combined status               (7)
control:  previous jump, movement direction/mode, previous combat,
          X charge progress, harpoon phase                             (6)
```

Player health and raw Boss HP are deliberately excluded from the observation.
Health loss and confirmed Boss damage remain available to the reward system.

The network is:

```text
24 inputs
  -> Linear(24, 96) + ReLU
  -> Linear(96, 96) + ReLU
  -> value head: 1
  -> joint-action advantage head: 53
```

This configuration has 16,950 trainable parameters.

## Curated Action Space

The executor retains three semantic fields:

```text
jump_z:   [released, press_z, hold_z]
movement: [neutral, hold_left, hold_right, left_dash, right_dash, harpoon_s]
combat:   [neutral, tap_x, hold_x, press_shift, up_x, down_x]
```

The Cartesian product would contain 108 vectors, but the policy exposes only
53 reviewed joint actions. The catalog applies these permanent rules:

- Harpoon S is atomic and exists only as `[0, 5, 0]`.
- A directed dash may be combined only with neutral combat or tap X.
- `press_z` may be combined only with neutral combat or hold X.
- Combining `press_z` with a directed dash permits only neutral combat.
- Generic facing-based dash and V are absent.
- Healing A is absent.

Every retained joint action has its own Q-value, so rewards remain attributed
to the complete simultaneous action rather than to independent branches.

Runtime masks further remove actions when dash, attack, quick cast, harpoon,
or charge continuation is unavailable. Replay stores the action actually
executed after masks and temporal locks.

## Temporal Actions

- Z represents release, press, or hold. The game decides whether that becomes
  a ground jump, double jump, or cloak hover.
- X charge must be held for at least 1,350 ms and is forcibly released at
  3,000 ms.
- S is blocked during X charge and for 500 ms after a completed charge release.
- Harpoon S owns a bounded 900 ms active/recovery lock and neutralizes the
  other fields during that lock.
- The policy selects a new action every 50 ms. Greedy jump and left/right
  movement retain a 200-300 ms commitment,
  which can be interrupted by danger, damage, or an invalid execution.

## Reward And Credit

Important current values are:

```text
step penalty:                         -0.001
attack-range entry:                   +0.2
confirmed Boss damage per HP:         +0.1
player damage per HP:                 -3.6
illegal action:                       -1.0
player parry event:                   +0.5
victory:                              +10.0
silk spent per unit:                  -0.04
successful Boss-attack evade budget:  +0.75
failed Boss-attack evade budget:      -1.0
combat threat-overlap responsibility: -0.75
normal/charged X miss:                -0.2
Shift miss:                           -0.8
five seconds without Boss damage:     -0.5
stagnation per later tick:            -0.025
one-time Boss proximity entry:        +0.05
arena boundary per tick:              -0.1
collision-risk increase:              up to -0.25
successful S hit bonus:               +50% of credited damage reward
successful S evade bonus:             +50% of the evade budget
```

Damage, parry, offensive-miss, player-hurt, and Boss-attack events can arrive
after the responsible input. Transitions therefore remain mutable in a pending
credit ledger for 40 control ticks. At 50 ms per tick this preserves the former
two-second attribution horizon. Only finalized immutable transitions enter replay.

A Boss attack remains open for 600 ms after its finish event so delayed player
hit telemetry can resolve the outcome. A successful evade distributes one
normalized `+0.75` budget across the complete action window. A failed evade
distributes one normalized `-1.0` budget.

Clearly unreachable attacks are masked. Predictive fringe attacks remain legal
without miss punishment. A miss penalty is applied only when the attack began
inside confirmed range, completed without interruption, and produced neither
Boss damage nor a parry during its result window.

## Exploration And Training

The shared replay buffer holds 50,000 transitions. Optimization starts after
2,000 transitions with batches of 128. The target network is updated every 1,000
training transitions.

Epsilon follows a mild reciprocal transition-based schedule from `0.60` to
`0.05` over 600,000 training transitions. Half of each sampled training batch
is randomly reflected left-to-right, including its action and legality mask, so
experience transfers between mirrored arena situations. Evaluation runs occur after every ten
training episodes with epsilon fixed to zero. Evaluation episodes do not alter
replay, optimizer state, global step, or the training episode count.

## Running

From the repository root:

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 1
```

Dry-run mode records proposed actions but does not train on inputs that were
not executed. Enable live keyboard control only when the game is ready:

```powershell
python -m hk_rl_DQN.real_dqn --episodes 10 --launch --execute-actions
```

Use `--full-window` to preserve the current game-window placement. Without it,
the trainer places the restored game window in the top-left quarter of the
desktop work area.

## Checkpoints

Checkpoint version 31 records:

- the `96 x 96` network shape;
- the complete ordered 53-action catalog;
- state, action, reward, and replay protocol versions;
- optimizer state, replay tensors, global step, and completed episodes.

Version 31 also separates `spin_attack` from `cyclone`, replaces the coarse
Boss-displacement bit with continuous collision risk, and uses 50 ms control.
Older checkpoints are intentionally incompatible. Start the new architecture
with:

```powershell
python -m hk_rl_DQN.real_dqn --reset
```

## Validation

Run the Python regression suite:

```powershell
python -m pytest hk_rl_DQN -q -p no:cacheprovider
```

Build the telemetry plugin:

```powershell
dotnet build hk_rl_DQN\external\karmelita_practice\KarmelitaPractice.csproj `
  --configuration Release
```

The curated-action tests verify the action count, stable IDs, removed
combinations, keyboard mappings, exploration output, and `(batch, 53)` network
shape.

## Source Layout

```text
real_state.py                         telemetry -> 24-value observation
real_reward.py                        immediate event rewards
real_dqn.py                           network, replay, delayed credit, trainer
final_project/action_executor.py      masks, timing, keyboard execution
final_project/action_catalog.py       atomic action compatibility catalog
external/karmelita_practice/          BepInEx telemetry plugin
tools/                                cold-start validation utilities
test_curated_actions.py               curated-action regression tests
```

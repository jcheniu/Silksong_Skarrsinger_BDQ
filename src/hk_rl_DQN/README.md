# Live Branching DQN

`hk_rl_DQN` is now the real-game Karmelita training pipeline. The old
single-action simulator implementation is archived under
`history/simulator_dqn_v1`.

The live pipeline is split by responsibility:

- `real_state.py`: 24-value observation encoding from telemetry. It contains
  10 motion values, normalized silk, seven compressed Boss semantics, and six
  previous-action/control values. Player health is deliberately absent from
  the observation for one-hit training, while health loss still drives reward.
- `real_reward.py`: event reward calculation. Confirmed hit events are derived
  from a decrease in the game's Boss `HealthManager.hp`, but raw Boss HP is
  never exposed to the DQN observation.
- `final_project/action_executor.py`: three independent BDQ action branches,
  legality masks, held-key timing, and JSONL action records.
- `real_dqn.py`: Branching Dueling Double DQN, replay, checkpoints, and the
  telemetry training loop.
- `tools/`: explicit cold-start state/action acceptance tests.

The fixed action vector is `[3,7,7]` in this order: `jump_z, movement, combat`.
`jump_z` is purely key based: release Z, press Z, or hold Z. Ground jump,
double jump, and cloak hover are not separate actions. `movement` chooses
neutral, held left/right, dash, a directed left/right dash, or S harpoon dash.
`combat` chooses neutral, tap X, hold X, press Shift, hold V, up+X, or down+X.
The three heads normally act simultaneously while values within one head remain
mutually exclusive. Harpoon dash is the deliberate exception: S is pulsed for
one tick, jump/combat are neutralized on launch, and all heads remain neutral
during its active/recovery lock. There is no dedicated wall-jump or sustained-run action.
Healing key `A` is deliberately absent from the action space and all input
adapters.
The full protocol and rationale are in
`final_project/BRANCHING_DQN_NOTES.md`.

The 24-value state layout is:

```text
motion:  player x/y, player vx/vy, relative boss x/y,
         relative boss-player vx/vy, grounded, player facing       (10)
resource: normalized silk                                           (1)
boss: behavior progress, attack category, aerial, displacement,
      vertical intent, hit pattern, combined status                 (7)
control: previous jump, movement direction/mode, previous combat,
         X charge progress, harpoon phase                            (6)
```

Relative velocity is `boss velocity - player velocity`; together with player
velocity it preserves both actors' motion while directly exposing closing speed.

Run a non-input pipeline check from the repository root:

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.real_dqn --episodes 1
```

Dry-run actions are logged, but they are not added to replay and do not update
the network because the corresponding keys were not executed in the game.

Add `--execute-actions` only when the game is ready for autonomous keyboard
control. Add `--launch` to let the trainer start the configured game executable.

Checkpoint behavior is conservative: without `--reset`, an existing checkpoint
is resumed and a missing checkpoint starts a new run. An incompatible or broken
existing checkpoint stops with an error and is never replaced implicitly. Use
`--reset` only when deliberately starting over; the flag clears model progress,
optimizer state, global step, completed-episode count, and replay memory.
Replay transitions are stored as compact tensors in the checkpoint, so a
restarted process resumes sampling immediately instead of repeating the
1,000-transition warmup. Checkpoints are written through a temporary file and
atomically replaced after serialization completes.

Telemetry exposes current/effective maximum silk, silk parts, skill cost,
ability-disable state, harpoon availability, and quick-cast availability.
`skill_s` is masked in the movement head using the game's
`CanHarpoonDash()` result and does not require silk;
`spell_shift` is masked when silk is insufficient or the action is unavailable.

The current reward protocol is `three-head-harpoon-movement-v14`.
Exploration starts at `0.60`, decays to `0.03` over 15,000 steps, and samples
sparse legal branch combinations instead of uniformly randomizing every branch.
Movement exploration weights left/right at 32% each, dash at 12%, directed
dashes at 8% each, and S at 8%. Exploratory left/right actions persist for a
total of 2-3 control ticks.
Boss attack types separated by less than 0.25 seconds form one hurt-sensitive
combo window. Damage is credited back to recent X/S/Shift start events. Dodge
credit trains the jump and movement heads. If the player loses health anywhere
in the complete attack/combo window, those same heads instead receive failed
dodge backfill of `-0.5 * 0.9^distance` on every buffered transition in the
window. The ordinary `-3.0` per lost player HP remains a common reward for all
heads. Taunt, silk, illegal-action, and offensive miss penalties train only
their responsible heads. X, completed charge releases, and Shift each receive
`-0.5` if their 20-tick result window expires without Boss HP loss or a
successful player parry. S damage trains the movement head, but an S movement
that deals no damage is not treated as an offensive miss. Telemetry counts
`HeroController.NailParry()` events. A new event inside a Boss attack/combo window
gives `+2.0` to the most recent X attack transition in the combat head. The
Boss `blocked` reaction is not used for this reward. Replay stores the executor's
actual action after smoothing and temporal fragments. BDQ computes a TD loss
for each active head instead of fitting only their mean. Use `--reset` because
checkpoint version 20 persists Replay Buffer contents alongside the compact
24-value state model.

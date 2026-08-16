# Live Joint-Action DQN

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
- `final_project/action_executor.py`: three coordinated action fields,
  legality masks, held-key timing, and JSONL action records.
- `real_dqn.py`: 147-action Dueling Double DQN, replay, checkpoints, and the
  telemetry training loop.
- `tools/`: explicit cold-start state/action acceptance tests.

The fixed action vector is `[3,7,7]` in this order: `jump_z, movement, combat`.
`jump_z` is purely key based: release Z, press Z, or hold Z. Ground jump,
double jump, and cloak hover are not separate actions. `movement` chooses
neutral, held left/right, dash, a directed left/right dash, or S harpoon dash.
`combat` chooses neutral, tap X, hold X, press Shift, press V, up+X, or down+X.
The three fields are combined into one of `3 * 7 * 7 = 147` joint actions and
receive one Q-value. Values within one field remain mutually exclusive.
Harpoon dash is the deliberate exception: S is pulsed for
one tick, jump/combat are neutralized on launch, and all fields remain neutral
during its active/recovery lock. There is no dedicated wall-jump or sustained-run action.
V is also atomic. The game FSM requires `CanCast`, no hard landing, and an
on-ground player. V uses a fixed 1,000 ms stationary recovery lock and launches
as `[0,0,4]`; jump and movement are neutral during launch and every recovery
tick.
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
control. During active training the restored game window is placed in the
top-left quarter of the primary desktop work area by default. Add
`--full-window` to preserve its existing size and position. Add `--launch` to
let the trainer start the configured game executable. Relaunched game windows
are positioned again automatically.

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

The current reward protocol is `normalized-evade-budget-v22-evaluation`.
Replay training starts at 1,000 transitions. The previous v19 schedule decayed
epsilon across 200 completed episodes, not 500. Version 20 instead drives
epsilon from training transitions: a normalized reciprocal curve falls quickly
at first and then slowly from `0.60` to `0.02` across 150,000 transitions.
Exploration samples sparse legal branch combinations instead of uniformly
randomizing every branch.
Movement exploration weights left/right at 32% each, dash at 12%, directed
dashes at 8% each, and S at 8%. Exploratory left/right actions persist for a
total of 2-3 control ticks. Combat exploration weights tap X, hold X, Shift,
taunt, up+X, and down+X as `30/8/8/1/20/20`, keeping taunt available without
letting it dominate cold-start data.
Player damage is `-3.6` per lost health point. That fixed event budget is
distributed across every pending transition in the two most recent contiguous
macro-action segments. Forced-neutral V ticks retain V ownership, so a sequence
ending in left movement then V penalizes the complete left and V segments.
Damage during V also applies its separate `-1.0` per-HP taunt-risk penalty to
the transition that launched V.
The plugin emits monotonic Boss `attack_id` events. An attack window starts at
the Boss attack-intent event and remains open for 700 ms after the finish event.
If the complete window is avoided, every action in the window receives part of
one fixed `+0.6` budget. Linear weights fall from `1.0` for the first action to
`0.5` for the final action and are normalized before distribution. Combat,
movement, jump, and neutral actions can all receive this credit, so attacking
and dodging rewards can stack. A hit instead applies one normalized `-1.0`
failure budget across the window. Player damage
applies one additional `-0.75`
combat responsibility budget only to started X/Shift actions whose short
recovery window overlapped an active Boss threat. It is not multiplied by HP.
X and completed charge releases receive `-0.25`, while Shift receives `-0.5`,
only when they started inside a confirmed vulnerable range, completed without
interruption, and their 20-tick result window expires without Boss HP loss or a
successful parry. S never receives an offensive-miss penalty.
The plugin's read-only `boss_vulnerable` flag comes directly from
`HealthManager.IsInvincible`; it controls masks and credit eligibility without
increasing the 24-value observation.
Out-of-range attacks are hard-masked; predictive fringe attacks remain legal
but are not treated as misses. S damage trains its joint action, but an S movement
that deals no damage is not treated as an offensive miss. Telemetry counts
`HeroController.NailParry()` events. A new event inside a Boss attack/combo window
gives `+0.8` to the most recent X attack transition. The
Boss `blocked` reaction is not used for this reward. Replay stores the executor's
actual action after smoothing and temporal fragments. Delayed outcomes mutate
only a pending-credit ledger; after the 20-tick attribution horizon, an
immutable scalar-reward transition is appended to replay. Unattributed damage
or parry events are reported but do not reinforce the current unrelated action.
Greedy jump and left/right movement receive a 200-300 ms minimum commitment.
An active/closing Boss attack, player damage, or an invalid executed action may
break the commitment early. Five seconds without Boss damage adds `-0.05`;
directional movement confined to 10% of arena width for about one second adds
`-0.05` per later tick. Entering the large outer Boss-proximity zone gives `+0.005` once,
then locks until confirmed Boss damage refreshes it and the player leaves and
re-enters. The outermost 10% at either arena boundary costs `-0.02` every tick.
Silk consumption costs `-0.04` per unit.
Use `--reset` because checkpoint version 28 normalizes successful evade credit;
version 27 replay must not be
reused.

X charge is valid for a release window rather than one fixed duration. It
becomes complete at 1,350 ms. With a 100 ms control tick, the first practical
release decision is at 1,400 ms. It may remain held until 3,000 ms and is
forced to release at the maximum. S is masked throughout the hold and for 500 ms after a
completed release, so harpoon movement cannot cancel the charge or its release
animation. Once charging starts, combat is locked to hold X until the 1,350 ms
minimum is reached; player damage can still interrupt it. Charge progress is
elapsed time normalized by the 3,000 ms maximum.

Every completed episode reports `replay_size`, `global_step`,
`gradient_updates`, `mean_loss`, the executor's `actual_action_counts` for
each action field, and one `mean_policy_q` for the selected joint action. A null loss with replay below
1,000 is expected because warmup has not completed. Loss is a moving TD target
and is not expected to decrease monotonically between episodes.
After every 10 training episodes, the next encounter is an independent greedy
evaluation with epsilon zero. Evaluation rows have `"evaluation": true`; they
do not append replay, update gradients, advance `global_step`, or increment the
training episode count.

Telemetry continues during encounter transitions and includes a monotonic
`encounter_id` for each plugin process. The trainer uses inactive snapshots or
a changed encounter ID to close and reopen the arena gate. A one-second
watchdog recovers a healthy new encounter if transition telemetry was missed.
When a game launched by the trainer exits normally before the requested run is
complete, the trainer saves replay, breaks the cross-process transition, and
relaunches it. Repeated rapid exits are capped at three per minute.

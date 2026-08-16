# Joint-Action DQN Design Notes

The executor keeps the semantic action vector `MultiDiscrete([3, 7, 7])`:

```text
jump_z:  [released, press_z, hold_z]
movement:[neutral, hold_left, hold_right, dash, left_dash, right_dash, harpoon_s]
combat:  [neutral, tap_x, hold_x, press_shift, hold_v, up_x, down_x]
```

The network does not optimize three independent heads. It enumerates all
`3 * 7 * 7 = 147` vectors and predicts one Q-value for each joint action. This
lets it learn that an attack can be useful while stationary but harmful during
a dodge, rather than assigning the attack the same value in both contexts.
Legality is still computed per semantic field and then expanded into a
147-value joint mask.

## Delayed Credit Ledger

Damage, parry, offensive-miss, taunt, player-hurt, and Boss-attack outcomes can
arrive several telemetry samples after the responsible action. New transitions
therefore remain in a pending ledger for 20 control ticks. Delayed events alter
only the pending reward. Finalization creates an immutable `Transition` and
appends it to replay; later events cannot mutate sampled training history.
Player damage is distributed across every pending tick in the two latest
contiguous macro-action segments. Forced temporal ticks retain their S or V
owner for this attribution.

Damage and parry rewards are assigned only when a compatible recent action
trial exists. Events with no defensible source remain visible in
`unattributed_damage_reward` or `unattributed_parry_reward`, but do not train an
unrelated current action.

## Boss Attack Credit

The telemetry plugin emits a monotonic `boss_attack.id` plus cumulative
started, active, finished, and player-hit event counters. A successful attack
window rewards every action from Boss intent through the post-finish grace
period:

```text
avoided window:                 +0.6 total
first action weight:            1.0
last action weight:             0.5
intermediate action weights:    linear interpolation, then normalization
player hit:                     -1.0 total
```

A finished attack remains unresolved for 700 ms so delayed
`last_player_hit_id` events can turn the complete combo into a failed evade.

Successful credit uses one normalized fixed budget: jump, movement, combat,
and neutral actions all receive their time-weighted share, allowing a safe
attack to stack offensive and evade reward without favoring longer Boss
attacks. The failure budget remains fixed across the whole attack window.
Player damage separately applies one
`-0.75` responsibility budget to
started X/Shift actions whose short recovery overlaps an active Boss threat.

Attack starts use three zones: unreachable actions are hard-masked, predictive
fringe actions are allowed without miss punishment, and only confirmed-range,
vulnerable, uninterrupted, completed actions can receive an offensive miss.
Normal and charged X misses cost `-0.25`, Shift misses cost `-0.5`, and S is
never penalized for missing.

## Temporal Actions

- A charge becomes successful at 1350 ms. With 100 ms ticks, the earliest
  practical release is 1400 ms and the maximum hold is 3000 ms.
- S is a movement-field harpoon dash. Its damage is credited normally, a miss
  is not penalized, and its displacement/recovery lock coordinates the other
  fields.
- S is suppressed during charge and for 500 ms after a completed release.
- V is accepted only when the plugin mirrors the real Taunt FSM checks:
  `CanCast()`, grounded, and not hard-landing. It receives a fixed 1,000 ms
  stationary executor lock.
- S and V are atomic joint actions. The 147-value legality mask excludes
  combinations that attach other fields to them, and exploration canonicalizes
  them before execution.
- Replay stores the executor's actual vector after masks and temporal locks,
  never merely the policy's attempted vector.

Greedy jump and left/right actions are committed for 200-300 ms unless danger,
damage, or an invalid execution requires an early break.

Checkpoint version 28 is incompatible with the earlier reward and action
semantics. Start older runs with `--reset`.

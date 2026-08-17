# Joint-Action DQN Design Notes

The executor keeps the semantic action vector `MultiDiscrete([3, 6, 6])`:

```text
jump_z:  [released, press_z, hold_z]
movement:[neutral, hold_left, hold_right, left_dash, right_dash, harpoon_s]
combat:  [neutral, tap_x, hold_x, press_shift, up_x, down_x]
```

The network does not optimize three independent heads. It predicts one Q-value
for each of 53 curated joint actions. This
lets it learn that an attack can be useful while stationary but harmful during
a dodge, rather than assigning the attack the same value in both contexts.
Legality is still computed per semantic field and then expanded into a
53-value joint mask. The catalog excludes dash with charge/Shift/directional X,
press-Z with tap X/Shift/directional X, and every non-atomic S combination.
The shared network is `24 -> 96 -> 96`, followed by one value output and 53
joint-action advantage outputs.

## Delayed Credit Ledger

Damage, parry, offensive-miss, player-hurt, and Boss-attack outcomes can
arrive several telemetry samples after the responsible action. New transitions
therefore remain in a pending ledger for 20 control ticks. Delayed events alter
only the pending reward. Finalization creates an immutable `Transition` and
appends it to replay; later events cannot mutate sampled training history.
Player damage is distributed across every pending tick in the two latest
contiguous macro-action segments. Forced temporal ticks retain their S owner
for this attribution.

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
avoided window:                 +0.75 total
first action weight:            1.0
last action weight:             0.5
intermediate action weights:    linear interpolation, then normalization
player hit:                     -1.0 total
```

A finished attack remains unresolved for 600 ms so delayed
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
Normal and charged X misses cost `-0.2`, Shift misses cost `-0.8`, and S is
never penalized for missing.

## Temporal Actions

- A charge becomes successful at 1350 ms. With 100 ms ticks, the earliest
  practical release is 1400 ms and the maximum hold is 3000 ms.
- S is a movement-field harpoon dash. Its damage is credited normally, a miss
  is not penalized, and its displacement/recovery lock coordinates the other
  fields.
- S is suppressed during charge and for 500 ms after a completed release.
- S is an atomic joint action. The catalog contains only `[0,5,0]`, and
  exploration canonicalizes it before execution.
- Replay stores the executor's actual vector after masks and temporal locks,
  never merely the policy's attempted vector.

Greedy jump and left/right actions are committed for 200-300 ms unless danger,
damage, or an invalid execution requires an early break.

Checkpoint version 30 is incompatible with the earlier network and action
semantics. Start older runs with `--reset`.

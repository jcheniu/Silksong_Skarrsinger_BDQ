# Live Branching DQN

`hk_rl_DQN` is now the real-game Karmelita training pipeline. The old
single-action simulator implementation is archived under
`history/simulator_dqn_v1`.

The live pipeline is split by responsibility:

- `real_state.py`: 46-value observation encoding from telemetry, including
  normalized current silk and jump hold/availability state.
- `real_reward.py`: event reward calculation. Confirmed hit events are derived
  from a decrease in the game's Boss `HealthManager.hp`, but raw Boss HP is
  never exposed to the DQN observation.
- `final_project/action_executor.py`: eight simultaneous key-state branches,
  legality masks, held-key timing, and JSONL action records.
- `real_dqn.py`: Branching Dueling Double DQN, replay, checkpoints, and the
  telemetry training loop.
- `tools/`: explicit cold-start state/action acceptance tests.

The fixed action vector is `[3,4,3,3,2,2,2,2]` in this order:
`horizontal, jump_z, dash_c, attack_x, skill_s, spell_shift, dream_d, taunt_v`.
`jump_z` separates short jump, held jump, and double jump; `dash_c` separates
dash and sustained ground sprint. The legacy `dream_d` slot is permanently masked to
neutral and never sends D. There is no dedicated wall-jump branch.
Healing key `A` is deliberately absent from the action space and all input
adapters.
The full protocol and rationale are in
`final_project/BRANCHING_DQN_NOTES.md`.

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
optimizer state, global step, and completed-episode count.

Telemetry exposes current/effective maximum silk, silk parts, skill cost,
ability-disable state, harpoon availability, and quick-cast availability.
`skill_s` uses the game's `CanHarpoonDash()` result and does not require silk;
`spell_shift` is masked when silk is insufficient or the action is unavailable.

The current reward protocol is `attack-window-credit-taunt-silk-v6`.
Exploration is decided once per complete branch vector and samples only legal
values. Replay stores the executor's actual action after smoothing and temporal
fragments. Successful dodge credit is discounted backward over the Boss attack
window. Silk spending costs `-0.02` per unit. V costs `-0.02` per held control
frame, plus `-0.5` when its six-frame outcome window expires without damage or
`-1` when the player is hurt. Player damage remains `-3` per lost health point.
Use `--reset` because checkpoint version 8 changes both state and reward semantics.

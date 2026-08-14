# Live Branching DQN

`hk_rl_DQN` is now the real-game Karmelita training pipeline. The old
single-action simulator implementation is archived under
`history/simulator_dqn_v1`.

The live pipeline is split by responsibility:

- `real_state.py`: 42-value observation encoding from telemetry, including
  normalized current silk.
- `real_reward.py`: event reward calculation; Boss HP is not used.
- `final_project/action_executor.py`: eight simultaneous key-state branches,
  legality masks, held-key timing, and JSONL action records.
- `real_dqn.py`: Branching Dueling Double DQN, replay, checkpoints, and the
  telemetry training loop.
- `tools/`: explicit cold-start state/action acceptance tests.

The fixed action vector is `[3,2,2,2,2,2,2,2]` in this order:
`horizontal, jump_z, dash_c, attack_x, skill_s, spell_shift, dream_d, taunt_v`.
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

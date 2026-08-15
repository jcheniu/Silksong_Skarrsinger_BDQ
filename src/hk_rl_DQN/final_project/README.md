# Final Real-Game Action Project

This directory is the isolated implementation area for the real Silksong
agent. It does not replace the verified `boss_env.py` simulator.

## Combat Actions

The executor consumes a three-value `jump_z`, `movement`, and `combat` vector
for each control tick. The DQN evaluates all 147 combinations jointly, then
decodes the selected ID back to this vector. This preserves combinations such
as `right + attack` and `left + jump` while learning their coordination value.
The recorder is intentionally separate from that adapter and only persists
decoded frames as JSONL.

The implemented Multi-Discrete executor and joint-action DQN are recorded in
`BRANCHING_DQN_NOTES.md`. The live trainer is `../real_dqn.py`; the former
single-action baseline is preserved under `history/simulator_dqn_v1`.

Boss attacks use plugin-emitted `attack_id` lifecycle events. Each completed
active attack distributes one fixed `+0.2` eligible-movement success or `-1.0`
failure budget across its pending joint actions. Replay insertion waits for delayed credit to
finish, so a sampled transition is immutable and cannot be rewritten later.

X attacks use three geometric zones. Clearly unreachable actions are masked,
predictive fringe actions remain legal without miss punishment, and only
confirmed vulnerable-range actions can receive a completed-animation miss.
Relative velocity predicts the Boss position 150 ms into the attack startup.
Successful player nail clashes are read from the telemetry
`player_parry_events` counter, not from the Boss `blocked` FSM reaction. A
clash inside the Boss attack window gives `+2.0` to the recent X attack action
in the combat head and counts as a productive attack outcome.

The action catalog is in `action_catalog.py`. It uses the local bindings:

- `LeftArrow` / `RightArrow`: move (100 ms minimum test hold)
- `Z`: generic release, press, or hold. The policy learns whether those key
  events produce a ground jump, double jump, or cloak hover from game state;
  there is no `wall_jump` action
- `LeftArrow` / `RightArrow` / `C` / `S`: one movement head with held
  direction, neutral/directed dash values, and harpoon dash
- `X`: tap for normal attack, or keep the `attack_charge` intent present over
  successive frames. Completion requires at least 1.35 s; with 100 ms control
  ticks, the first practical release is at 1.4 s. It is forced to release at
  3.0 s. S is blocked during the hold and for 500 ms after a
  completed release. Only a release at or after 1.35 s creates attack credit;
  the executor keeps X held until that minimum, while a hit interrupts and
  resets it.
- `UpArrow + X` / `DownArrow + X`: upward and downward attacks are separate
  combat-head values. They create the same X attack-start credit event as a
  normal attack.
- `LeftShift`: quick cast; legality uses current silk, skill cost, ability state,
  and the game's control/cooldown result
- `S`: harpoon dash (`KeySupDash`); legality uses `CanHarpoonDash()` and does
  not depend on current silk. It belongs to the movement head because its
  displacement and recovery dominate its tactical effect. Boss damage is
  credited to the movement transition, but an S movement that deals no damage
  is not treated as an offensive miss. Holding S is bounded to 900 ms.
- `D`: disabled and never sent by the live policy
- `V`: battle taunt. Live training applies a small held cost and a delayed
  penalty to the combat-head transition that selected V. Boss damage is not
  treated as proof that taunting
  succeeded. Sparse exploration keeps V available without selecting it in
  half the frames.

Healing key `A` is disabled. `Q`, `I`, `Tab`, `J`, and `T` are also
intentionally excluded.

## Recording Actions

Record one command as JSONL:

```powershell
python -m hk_rl_DQN.final_project.action_recorder quick_cast --duration-ms 80
python -m hk_rl_DQN.final_project.action_recorder harpoon_dash --duration-ms 80
python -m hk_rl_DQN.final_project.action_recorder attack_charge --duration-ms 1350
```

The recorder rejects durations shorter than the action's minimum hold time.
It records silk consumption as metadata; the real adapter must still reject an
action when telemetry reports insufficient silk.

Every catalog action also has a same-named callable in `action_functions.py`.
`tools/run_all_actions.py` uses the same semantic-vector executor as live training.
It covers every non-neutral branch value, directed dashes, charge release, and
one simultaneous joint-action combination, then writes a `summary.json` plus one
JSONL file per case.
Use `--interval-s 1` (the default) to leave a one-second gap between repeated
tests. With `--reuse-game`, the same game process is retained while the native
death-restart loop prepares the next attempt.

## One-Action Cold Start

Run from the repository root with `PYTHONPATH=src`. The test starts a fresh
game process, tails the log in real time, waits for the challenge marker, then
sends exactly one key. The default marker is the currently verified
`Karmelita challenge state: Challenge Complete`. Replace it with a precise
war-cry-ended telemetry marker when that event is available.

```powershell
$env:PYTHONPATH='src'
python -m hk_rl_DQN.tools.cold_start_action_test attack_charge `
  --duration-ms 500 `
  --game-exe 'C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight Silksong\Hollow Knight Silksong.exe' `
  --log 'C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight Silksong\BepInEx\LogOutput.log'
```

The exact live-policy vector can also be tested directly. This example applies
`hold Z + right dash + tap X` in one control tick:

```powershell
python -m hk_rl_DQN.tools.cold_start_action_test `
  --action-vector 2 5 1 --ticks 1 --tick-ms 100 `
  --game-exe 'C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight Silksong\Hollow Knight Silksong.exe' `
  --log 'C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight Silksong\BepInEx\LogOutput.log'
```

Use one cold start per action case. The default batch order covers jump press
and hold, movement, all dash values, combat values, then the simultaneous
combination. The script intentionally pauses after the vector is sent so the
operator can inspect whether resources changed, animations started, and the
boss was affected.

## Cold-Start Test Protocol

Do not automatically inject the recorded actions yet. For each action:

1. Start Silksong cold with the current Karmelita plugin.
2. Wait for `Karmelita challenge state: Challenge Complete`.
3. Observe the boss's end-of-war-cry/recovery state in telemetry/logs.
4. Execute exactly one recorded action.
5. Record whether the action started, consumed silk, hit the boss, or was
   rejected by cooldown/state.
6. Let the encounter reset, then repeat with the next action.

The action should be enabled for DQN only after its result is confirmed in the
game. `LogOutput.log` is for acceptance evidence; JSONL telemetry is the
authoritative action/state stream.

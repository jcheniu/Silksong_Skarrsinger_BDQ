# Final Real-Game Action Project

This directory is the isolated implementation area for the real Silksong
agent. It does not replace the verified `boss_env.py` simulator.

## Combat Actions

The policy output is a multi-hot tensor/list for each control tick, not one
exclusive action. A downstream adapter reads that frame and holds all selected
keys together for the tick, so `right + attack` and `left + jump` are valid.
The recorder is intentionally separate from that adapter and only persists
decoded frames as JSONL.

The implemented Multi-Discrete / Branching DQN architecture is recorded in
`BRANCHING_DQN_NOTES.md`. The live trainer is `../real_dqn.py`; the former
single-action baseline is preserved under `history/simulator_dqn_v1`.

The action catalog is in `action_catalog.py`. It uses the local bindings:

- `LeftArrow` / `RightArrow`: move (100 ms minimum test hold)
- `Z`: short jump, sustained hold-jump fragment, and a separate double-jump
  pulse; there is no `wall_jump` action
- `C`: separate dash and sustained-ground-run intents
- `X`: tap for normal attack, or keep the `attack_charge` intent present over
  successive frames. The charge state accumulates toward 1.35 s (81 frames);
  omitting it, switching/releasing, or a hit interrupts and resets it.
- `LeftShift`: quick cast; legality uses current silk, skill cost, ability state,
  and the game's control/cooldown result
- `S`: harpoon dash (`KeySupDash`); legality uses `CanHarpoonDash()` and does
  not depend on current silk
- `D`: disabled and never sent by the live policy
- `V`: battle taunt. Live training applies a small held cost and evaluates a
  six-tick hit/hurt outcome window, so repeated unproductive taunts are penalized.

Healing key `A` is disabled. `Q`, `I`, `Tab`, `J`, and `T` are also
intentionally excluded.

## Recording Actions

Record one command as JSONL:

```powershell
python -m hk_rl_DQN.final_project.action_recorder quick_cast --duration-ms 80
python -m hk_rl_DQN.final_project.action_recorder harpoon_dash --duration-ms 80
python -m hk_rl_DQN.final_project.action_recorder quick_run --duration-ms 600
python -m hk_rl_DQN.final_project.action_recorder attack_charge --duration-ms 1350
```

The recorder rejects durations shorter than the action's minimum hold time.
It records silk consumption as metadata; the real adapter must still reject an
action when telemetry reports insufficient silk.

Every catalog action also has a same-named callable in `action_functions.py`.
`tools/run_all_actions.py` runs those actions one at a time in independent cold
starts and writes a `summary.json` plus one JSONL file per action.
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

Use one cold start per action. Recommended observation order is
`quick_cast`, `harpoon_dash`, `quick_run`, `attack_charge`, then
`taunt` and `dreamnail`. The script intentionally pauses after the key is sent
so the operator can inspect whether silk decreased, the action animation
started, and whether the boss was affected.

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

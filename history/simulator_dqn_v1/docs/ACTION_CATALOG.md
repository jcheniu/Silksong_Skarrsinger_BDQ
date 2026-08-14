# Hornet Action Catalog

This is the single action vocabulary used by the DQN and the real-game
adapter. IDs are stable and start at zero.

| ID | Action | Current status |
|---:|---|---|
| 0 | `left` | Implemented in Python simulator |
| 1 | `right` | Implemented in Python simulator |
| 2 | `dash` | Implemented in Python simulator; verify cooldown in game |
| 3 | `attack` | Implemented in Python simulator; needle attack in game |
| 4 | `wait` | Implemented in both as no intentional input |
| 5 | `jump` | Implemented in simulator; game may produce wall-jump behavior |
| 6 | `attack_charge` | Hold `X`; real-game candidate |
| 7 | `cast` | `A`; spell/cast, requires silk and equipped ability |
| 8 | `super_dash` | `S`; requires a valid charge state |
| 9 | `quick_run` | Hold `C`; fast run |
| 10 | `quick_cast` | Left `Shift`; quick cast/skill |
| 11 | `dreamnail` | `D`; contextual interaction |
| 12 | `taunt` | `V`; battle taunt |
| 13 | `wall_jump` | `Z` + direction; requires wall contact |

The simulator currently exposes IDs 0-5 to preserve compatibility with the
existing six-output checkpoint. IDs 6 onward are intentionally runtime-only until
telemetry confirms their actual input binding, cooldown, and effect in the
Karmelita arena. Enabling them in the DQN output before that validation would
teach the agent actions the environment cannot execute.

Verified local bindings come from `HKCU\Software\Team Cherry\Hollow Knight
Silksong`: `Z` jump, `X` attack, `C` dash, `A` cast, `S` super dash, `D`
dreamnail, left/right arrows for movement, and `LeftShift` quick cast. Up/down
arrows only look vertically and are excluded from the combat action space. A held `X` is
represented as `attack_charge`; a held `C` is represented as `quick_run`.
Menu-only inputs (`Q`, `I`, `Tab`, `J`, `T`) are intentionally excluded from
the DQN action space. `V` is retained as a battle action.

The real adapter writes commands to JSONL using `real_action_protocol.py`.
Each command includes an action ID, action name, timestamp, duration, and a
`hold` flag. Use `attack` with a short duration for a tap of `X`; use
`attack_charge` with a longer duration to hold `X` for a charged attack.

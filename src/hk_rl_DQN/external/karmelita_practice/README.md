# Karmelita Practice Loop

Loads save slot 1 and immediately enters the original Skarrsinger Karmelita
memory encounter. Hornet is restored to full health and the complete encounter
is reloaded after each death. `F8` remains available as a manual reload.
The pre-boss enemy wave is ended immediately and Hornet is placed directly in
the boss room during the game's scene-ready callback, before normal control is
restored.

## Disable Or Re-enable

Close the game and edit:

`BepInEx\config\io.github.hollow-knight-rl.karmelita-practice.cfg`

Set `Enabled = false` under `[General]` to disable all plugin behavior. Set it
back to `true` and restart the game to re-enable it. When disabled, the plugin
does not patch the game, load a save, enter the arena, handle `F8`, or restart
after death.

As a hard fallback, close the game and rename `KarmelitaPractice.dll` to
`KarmelitaPractice.dll.disabled`. Rename it back to re-enable the plugin.

The verified encounter identifiers are based on the MIT-licensed public source
of `MicheliniDev/KarmelitaPrime`: `Memory_Ant_Queen`, `door_wakeInMemory`,
`defeatedAntQueen`, and `Boss Scene/Hunter Queen Boss`. This project does not
include KarmelitaPrime's difficulty, visual, audio, or FSM modifications.

## Read-only Telemetry Snapshot

The plugin writes a JSONL snapshot stream while it is running. The default
file is:

`BepInEx/plugins/hollow-knight-rl-KarmelitaPractice/telemetry.jsonl`

Each snapshot contains the active scene, frame/timestamp, encounter status and
`encounter_id`,
player and Boss position/velocity, player health/resources, cumulative player
`NailParry()` events, and the Boss-tree and challenge PlayMaker FSMs with their
hierarchy path, name, and current state.
`boss_vulnerable` is the inverse of the Boss `HealthManager.IsInvincible`
flag. It gates new X attack starts and miss eligibility but is not added as a
new DQN observation dimension.
The `boss_attack` object provides a monotonic attack ID, semantic type and
phase, cumulative start/active/finish/player-hit counters, and the most recent
finished and player-hit IDs. The trainer uses these lifecycle events to settle
one fixed dodge budget per attack instead of inferring credit from sample count.
Snapshots continue while the plugin is enabled but the encounter is inactive
or transitioning. `encounter_id` increments whenever a newly loaded Boss
encounter becomes active, allowing the trainer to distinguish a fresh fight
from lingering terminal frames even if an inactive sample is missed.
Sampling defaults to 50 ms so state updates match the live controller tick.
The settings are generated in the plugin config:

```json
{
  "player_resources": {
    "silk": 6,
    "silk_max": 9,
    "silk_parts": 2,
    "skill_cost": 4,
    "silk_abilities_disabled": false,
    "skill_available": true,
    "spell_available": true
  },
  "player_control": {
    "jump_available": true,
    "dash_available": true,
    "attack_available": true
  }
}
```

`skill_available` is the result of `HeroController.CanHarpoonDash()`.
`spell_available` combines current silk, `PlayerData.SilkSkillCost`, the
silk-ability disable flag, and `HeroController.CanThrowTool(false)`.

```ini
[Telemetry]
Enabled = true
IntervalSeconds = 0.05
```

The recorder remains read-only. It observes FSM state and game events without
changing Boss health, FSM variables, hitboxes, or encounter behavior.

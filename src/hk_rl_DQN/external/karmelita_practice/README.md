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

Each snapshot contains the active scene, frame/timestamp, encounter status,
player and Boss position/velocity, player health/resources, and the Boss-tree and
challenge PlayMaker FSMs with their hierarchy path, name, and current state.
Sampling defaults to 100 ms. The settings are generated in the plugin config:

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
IntervalSeconds = 0.1
```

This first recorder is intentionally read-only. Boss health, FSM variables,
attack hitboxes, and action-level event labels are not inferred until the
snapshot stream has been inspected in a live encounter.

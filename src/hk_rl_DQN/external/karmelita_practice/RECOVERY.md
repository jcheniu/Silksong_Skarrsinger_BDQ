# Karmelita Practice Recovery

This document restores the verified local setup that starts Silksong, loads
save slot 1, enters `Memory_Ant_Queen`, triggers Karmelita automatically, keeps
the camera on Hornet and Karmelita, and reloads a clean save after every death.

> 中文结论：卸载或重装游戏通常会删除游戏目录内的 BepInEx 和已部署
> 插件，所以当前效果会暂时失效；工作区源码和 `AppData\LocalLow` 中的
> 存档通常不会被 Steam 删除。按本文恢复 BepInEx、存档和插件后即可复现。

## What A Reinstall Removes

A Steam uninstall/reinstall can remove everything below the game directory,
including:

- `winhttp.dll`, `.doorstop_version`, and `doorstop_config.ini`
- the complete `BepInEx` directory
- the deployed `KarmelitaPractice.dll`

It does not normally remove the checked-out source directory:

`<repo>\src\hk_rl_DQN\external\karmelita_practice`

It also does not normally remove saves under `AppData\LocalLow`, but back up
the save before uninstalling:

`%USERPROFILE%\AppData\LocalLow\Team Cherry\Hollow Knight Silksong\1182828463\user1.dat`

Run this before uninstalling to keep a recovery copy beside this document:

```powershell
$sourceSave = Join-Path $env:USERPROFILE 'AppData\LocalLow\Team Cherry\Hollow Knight Silksong\1182828463\user1.dat'
$backupSave = Join-Path (Split-Path (Resolve-Path '.').Path) 'hollow-knight-rl-local-backup\user1.dat'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backupSave) | Out-Null
Copy-Item -LiteralPath $sourceSave -Destination $backupSave -Force
Get-FileHash -LiteralPath $sourceSave,$backupSave -Algorithm SHA256
```

The two printed hashes must match. The local backup directory contains
personal save data and should not be committed or shared.

## Verified Baseline

- Game directory: `$steamRoot\Hollow Knight Silksong` (where `$steamRoot` is your Steam library path)
- Unity runtime: `6000.0.50.1503346`
- BepInEx: `5.4.23.4`
- Plugin: `KarmelitaPractice 0.1.0`
- Save: slot 1 (`user1.dat`)
- Build dependency: `Silksong.GameLibs 1.2.0-silksong1.0.29315`
- Verified DLL SHA-256:
  `C8CE2A482C9104B1BE3377B0FD91B46A030256488FFEDCA2EF073C65264CF464`

The hash applies only to the currently verified DLL. A source rebuild may
produce a different hash even when behavior is equivalent.

## Recovery Procedure

1. Install Silksong through Steam and launch it once without mods. Close it.
2. Install the official x64 BepInEx 5.4.23.4 package from
   `https://github.com/BepInEx/BepInEx/releases/tag/v5.4.23.4` into the game
   directory. Use `BepInEx_win_x64_5.4.23.4.zip`. The
   game directory must directly contain `winhttp.dll`, `doorstop_config.ini`,
   and `BepInEx`.
3. Launch the game once, wait for the main menu, then close it. Confirm this
   file exists:

   `$gameDirectory\BepInEx\LogOutput.log`

4. Restore `user1.dat` to the save path above if Steam Cloud did not restore
   slot 1. With the backup command above, restore it using:

   ```powershell
   $backupSave = Join-Path (Split-Path (Resolve-Path '.').Path) 'hollow-knight-rl-local-backup\user1.dat'
   $saveDirectory = Join-Path $env:USERPROFILE 'AppData\LocalLow\Team Cherry\Hollow Knight Silksong\1182828463'
   New-Item -ItemType Directory -Force -Path $saveDirectory | Out-Null
   Copy-Item -LiteralPath $backupSave -Destination (Join-Path $saveDirectory 'user1.dat') -Force
   ```

5. Restore the plugin using either Method A or Method B.

### Method A: Copy The Verified DLL

Create this directory:

`$gameDirectory\BepInEx\plugins\hollow-knight-rl-KarmelitaPractice`

Copy the built DLL from `<repo>\src\hk_rl_DQN\external\karmelita_practice\bin\Debug\netstandard2.1\KarmelitaPractice.dll` into it.

PowerShell equivalent (run PowerShell as Administrator if access is denied):

```powershell
$pluginDirectory = Join-Path ${env:ProgramFiles(x86)} 'Steam\steamapps\common\Hollow Knight Silksong\BepInEx\plugins\hollow-knight-rl-KarmelitaPractice'
$verifiedDll = Join-Path (Resolve-Path '.') 'src\hk_rl_DQN\external\karmelita_practice\bin\Debug\netstandard2.1\KarmelitaPractice.dll'
New-Item -ItemType Directory -Force -Path $pluginDirectory | Out-Null
Copy-Item -LiteralPath $verifiedDll -Destination (Join-Path $pluginDirectory 'KarmelitaPractice.dll') -Force
```

Optional integrity check:

```powershell
Get-FileHash `
  (Join-Path $pluginDirectory 'KarmelitaPractice.dll') `
  -Algorithm SHA256
```

### Method B: Rebuild And Deploy From Source

Confirm `SilksongPath.props` points to the new installation directory. Then:

```powershell
Set-Location (Join-Path (Resolve-Path '.') 'src\hk_rl_DQN\external\karmelita_practice')
dotnet restore KarmelitaPractice.csproj
dotnet build KarmelitaPractice.csproj --no-restore -c Debug
```

The project automatically deploys the DLL to the BepInEx plugin directory.
Expected build result: `0 warnings`, `0 errors`.

## Cold-Start Acceptance Check

Start the game normally and do not select a save. The main menu is briefly
visible, then the plugin loads slot 1 and enters the fight. In
`BepInEx\LogOutput.log`, require all of these lines:

```text
Loading [KarmelitaPractice 0.1.0]
Automatically loading save slot 1
Save slot 1 loaded; continuing game
Entering Skarrsinger Karmelita
Karmelita boss found: Hunter Queen Boss
Karmelita challenge state: Challenge Complete
Karmelita ground: Chunk 0 4 at (155.00, 18.00), heroY=19.57
Karmelita arena bounds: 134.43 to 162.67
Karmelita settled position: Hornet=(155.95, 19.57, 0.00), Scene=Memory_Ant_Queen
Automatic Karmelita entry complete
```

Kill Hornet once. Require this second sequence:

```text
Hornet died; restarting Karmelita encounter
Death flow settled in Song_Enclave; reloading save slot 1
Save slot 1 reloaded cleanly in Song_Enclave; entering challenge
Entering Skarrsinger Karmelita
Karmelita challenge state: Challenge Complete
Karmelita settled position: Hornet=(155.95, 19.57, 0.00), Scene=Memory_Ant_Queen
```

Do not accept a recovery that contains `ScenePreloader.Cleanup`,
`Scene to unload is invalid`, a missing `Hunter Queen Boss`, or a settled scene
other than `Memory_Ant_Queen`.

## If The Game Was Updated

Steam may install a newer Silksong build during reinstall. If the plugin no
longer loads or the expected scene objects differ:

1. Keep the source and save backup unchanged.
2. Reinstall the current compatible BepInEx Silksong pack.
3. Update `Silksong.GameLibs` in `KarmelitaPractice.csproj` to the package that
   matches the installed game build.
4. Rebuild and run the cold-start acceptance check.
5. If scene identifiers changed, revalidate these constants before editing
   other logic: `Memory_Ant_Queen`, `door_wakeInMemory`,
   `Boss Scene/Hunter Queen Boss`, and `defeatedAntQueen`.

Do not delete the source project after deployment. The DLL inside the game
directory is disposable; this directory is the recovery source of truth.

## Recovery Guarantee Boundary

For the verified game/runtime versions above, the source project, verified
DLL, save backup, and acceptance logs are sufficient to reproduce the current
setup after a clean game reinstall. A reinstall that also changes the game
runtime or scene data is a compatibility migration, not an identical restore;
complete the update procedure and all acceptance checks before treating it as
recovered.

## Disable Or Re-enable The Plugin

Close the game and edit:

`$gameDirectory\BepInEx\config\io.github.hollow-knight-rl.karmelita-practice.cfg`

Under `[General]`, use:

```ini
Enabled = false
```

Restart the game. The disabled plugin does not install its scene patches, load
save slot 1, react to `F8`, enter Karmelita, or intercept the death flow. To
restore the practice loop, set `Enabled = true` and restart the game.

Hard fallback: while the game is closed, rename the deployed
`KarmelitaPractice.dll` to `KarmelitaPractice.dll.disabled`. BepInEx will not
load it. Rename it back to `KarmelitaPractice.dll` to restore it. Do not delete
the verified DLL in the workspace.

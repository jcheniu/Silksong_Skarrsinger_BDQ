# Print the local Silksong keyboard bindings from Unity's registry storage.
# This is read-only and is intended to be rerun after changing game controls.
$keyPath = 'Software\Team Cherry\Hollow Knight Silksong'
$names = @(
  'KeyUp', 'KeyDown', 'KeyLeft', 'KeyRight', 'KeyJump', 'KeyAttack',
  'KeyDash', 'KeyCast', 'KeySupDash', 'KeyDreamnail', 'KeyQuickMap',
  'KeyQuickCast', 'KeyTaunt', 'KeyInventory', 'KeyInventoryTools',
  'KeyInventoryJournal', 'KeyInventoryQuests'
)

$key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($keyPath)
if ($null -eq $key) {
  throw "Silksong registry key not found: HKCU\$keyPath"
}

$allValues = $key.GetValueNames()
foreach ($name in $names) {
  $valueName = $allValues | Where-Object { $_ -like "${name}_*" } | Select-Object -First 1
  if ($null -eq $valueName) {
    [pscustomobject]@{ Action = $name; Binding = 'MISSING' }
    continue
  }

  $raw = $key.GetValue($valueName, $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
  if ($raw -is [byte[]]) {
    $binding = [Text.Encoding]::UTF8.GetString($raw).Trim([char]0)
  } else {
    $binding = [string]$raw
  }
  [pscustomobject]@{ Action = $name; Binding = $binding }
}

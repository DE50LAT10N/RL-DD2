param(
  [string]$GameRoot = "",
  [switch]$Build,
  [switch]$DumpData,
  [switch]$SkipHashVerify
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$slnPath = Join-Path $PSScriptRoot "DdRL.sln"
$srcDir = Join-Path $PSScriptRoot "DdRL.Plugin\bin\Release\netstandard2.1"
$srcDll = Join-Path $srcDir "DdRL.Plugin.dll"
$srcNewtonsoft = Join-Path $srcDir "Newtonsoft.Json.dll"

function Resolve-GameRoot {
  param([string]$RequestedRoot)

  if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
    return (Resolve-Path -LiteralPath $RequestedRoot -ErrorAction Stop).Path
  }

  $running = Get-Process "Darkest Dungeon II" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($running -and $running.Path) {
    return (Split-Path -Parent $running.Path)
  }

  $steamCommonRoots = @(
    "D:\SteamLibrary\steamapps\common",
    "C:\Program Files (x86)\Steam\steamapps\common",
    "C:\Program Files\Steam\steamapps\common"
  )

  foreach ($root in $steamCommonRoots) {
    $candidate = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -like "Darkest Dungeon* II" -and (Test-Path (Join-Path $_.FullName "Darkest Dungeon II.exe")) } |
      Select-Object -First 1
    if ($candidate) {
      return $candidate.FullName
    }
  }

  throw "GameRoot was not provided and DD2 install folder could not be discovered. Pass -GameRoot explicitly."
}

$GameRoot = Resolve-GameRoot $GameRoot
$dstDir = Join-Path $GameRoot "BepInEx\plugins\DdRL"
$dstDll = Join-Path $dstDir "DdRL.Plugin.dll"

if ($Build) {
  dotnet build $slnPath -c Release | Out-Host
}

if (!(Test-Path $srcDll)) {
  throw "Build output not found: $srcDll"
}

New-Item -ItemType Directory -Path $dstDir -Force | Out-Null

# Fail-fast lock check on primary plugin DLL.
if (Test-Path $dstDll) {
  try {
    Remove-Item $dstDll -Force -ErrorAction Stop
  } catch {
    throw "Cannot replace locked plugin DLL: $dstDll`nClose the game and retry."
  }
}

try {
  Copy-Item -Path (Join-Path $srcDir "*.dll") -Destination $dstDir -Force -ErrorAction Stop
} catch {
  throw "Cannot copy plugin runtime DLLs into $dstDir.`nClose Darkest Dungeon II and retry install.`n$($_.Exception.Message)"
}
if (Test-Path $srcNewtonsoft) {
  try {
    Copy-Item -Path $srcNewtonsoft -Destination (Join-Path $dstDir "Newtonsoft.Json.dll") -Force -ErrorAction Stop
  } catch {
    throw "Cannot copy Newtonsoft.Json.dll into $dstDir.`nClose Darkest Dungeon II and retry install.`n$($_.Exception.Message)"
  }
} else {
  Write-Warning "Newtonsoft.Json.dll was not found at $srcNewtonsoft"
}

if (-not $SkipHashVerify) {
  $srcHash = (Get-FileHash $srcDll -Algorithm SHA256).Hash
  if (!(Test-Path $dstDll)) {
    throw "Install failed: destination DLL is missing: $dstDll"
  }
  $dstHash = (Get-FileHash $dstDll -Algorithm SHA256).Hash
  if ($srcHash -ne $dstHash) {
    throw "Hash mismatch after install. Source=$srcHash Destination=$dstHash"
  }
  Write-Host "Hash verified: $dstHash"
}

Write-Host "Game root: $GameRoot"
Write-Host "Installed DdRL plugin and runtime dependencies -> $dstDir"
Write-Host "Installed plugin DLL: $dstDll"

if ($DumpData) {
  $cfgPath = Join-Path $GameRoot "BepInEx\config\com.rl.ddrl.cfg"
  if (Test-Path $cfgPath) {
    (Get-Content $cfgPath) `
      -replace "Debug.EnableDataDump = false", "Debug.EnableDataDump = true" `
      -replace "Debug.DumpOnlyExit = false", "Debug.DumpOnlyExit = true" `
      | Set-Content $cfgPath
  }
  Write-Host "DumpData requested. Launch DD2 once; plugin will dump and exit."
}

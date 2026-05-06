param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8765,
  [string]$Model = "runs/best/best_model.zip",
  [ValidateSet("ppo", "pass_only", "hook_only")]
  [string]$Mode = "ppo",
  [int]$MaxSteps = 200,
  [double]$Timeout = 6.0,
  [double]$ResetTimeout = 120.0,
  [double]$ActionTimeout = 10.0,
  [double]$EnemyTurnWait = 60.0,
  [double]$StunnedTurnWait = 12.0,
  [double]$StepDelay = 0.25,
  [int]$ServerCheckRetries = 5,
  [string]$ExpectedModVersion = "0.1.21-stability",
  [switch]$DisablePassActions,
  [switch]$AllowPassActions,
  [switch]$DisableEmergencyPass,
  [switch]$AllowPolicyMoveActions,
  [string]$Python = "",
  [string[]]$GameProcessNames = @("Darkest Dungeon II", "DarkestDungeonII", "DarkestDungeon2", "Darkest Dungeon 2"),
  [switch]$Stochastic,
  [switch]$Quiet,
  [switch]$SkipProcessCheck,
  [switch]$AllowOutOfBattle
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-Python {
  param([string]$Requested, [string]$RepoRoot)

  if (-not [string]::IsNullOrWhiteSpace($Requested)) {
    if (-not (Get-Command $Requested -ErrorAction SilentlyContinue)) {
      throw "Python executable was not found: $Requested"
    }
    return $Requested
  }

  $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
  if (Test-Path $venvPython) {
    return $venvPython
  }

  if (Get-Command "python" -ErrorAction SilentlyContinue) {
    return "python"
  }

  throw "Python was not found. Pass -Python explicitly or create .venv."
}

function Assert-GameRunning {
  param([string[]]$Names)

  $processes = Get-Process -ErrorAction SilentlyContinue
  foreach ($name in $Names) {
    $needle = $name.Trim()
    if ([string]::IsNullOrWhiteSpace($needle)) {
      continue
    }

    $matches = $processes | Where-Object {
      $_.ProcessName -ieq $needle -or
      $_.ProcessName -like "*$needle*" -or
      ($_.MainWindowTitle -and $_.MainWindowTitle -like "*$needle*")
    }
    if ($matches) {
      $first = $matches | Select-Object -First 1
      Write-Host "Game process found: $($first.ProcessName) (pid=$($first.Id))"
      return
    }
  }

  throw "Darkest Dungeon II process was not found. Start the game first or pass -SkipProcessCheck."
}

function Read-NdjsonLine {
  param(
    [System.Net.Sockets.TcpClient]$Client,
    [System.IO.StreamReader]$Reader,
    [DateTime]$Deadline
  )

  while ([DateTime]::UtcNow -lt $Deadline) {
    if ($Client.Available -gt 0) {
      $line = $Reader.ReadLine()
      if (-not [string]::IsNullOrWhiteSpace($line)) {
        return $line
      }
    }
    Start-Sleep -Milliseconds 50
  }
  return $null
}

function Test-ModServer {
  param(
    [string]$HostName,
    [int]$Port,
    [double]$Timeout,
    [int]$Retries,
    [string]$ExpectedModVersion,
    [bool]$RequireBattle
  )

  $lastError = $null
  for ($attempt = 1; $attempt -le [Math]::Max(1, $Retries); $attempt++) {
  $client = [System.Net.Sockets.TcpClient]::new()
  $reader = $null
  try {
    $connect = $client.BeginConnect($HostName, $Port, $null, $null)
    if (-not $connect.AsyncWaitHandle.WaitOne([TimeSpan]::FromSeconds($Timeout))) {
      throw "Connection timed out"
    }
    $client.EndConnect($connect)
    $client.ReceiveTimeout = [int]($Timeout * 1000)
    $client.SendTimeout = [int]($Timeout * 1000)

    $stream = $client.GetStream()
    $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)

    $requestId = 4242
    $payload = "{""type"":""ping"",""request_id"":$requestId}`n"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Flush()

    $deadline = [DateTime]::UtcNow.AddSeconds($Timeout)
    $sawPong = $false
    $lastState = $null
    $hasLiveBattleState = $false
    $modVersion = $null

    while ([DateTime]::UtcNow -lt $deadline) {
      $line = Read-NdjsonLine -Client $client -Reader $reader -Deadline $deadline
      if ($null -eq $line) {
        break
      }

      $msg = $line | ConvertFrom-Json
      if ($msg.type -eq "hello") {
        $modVersion = [string]$msg.mod_version
        continue
      }
      if ($msg.type -eq "pong" -and [int]$msg.request_id -eq $requestId) {
        $sawPong = $true
        continue
      }
      if ($msg.type -eq "state") {
        $lastState = $msg
        $heroes = @($msg.heroes)
        $enemies = @($msg.enemies)
        if ($heroes.Count -gt 0 -and $enemies.Count -gt 0 -and (-not $RequireBattle -or [bool]$msg.in_battle)) {
          $hasLiveBattleState = $true
        }
      }
    }

    if (-not $sawPong) {
      throw "Server connected, but ping/pong did not complete."
    }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedModVersion)) {
      if (-not $modVersion) {
        throw "DDRL plugin hello.mod_version was not received. Close DD2, reinstall the plugin, and launch DD2 again."
      }
      if ($modVersion -ne $ExpectedModVersion) {
        throw "Loaded DDRL plugin version is '$modVersion', expected '$ExpectedModVersion'. Close the game, reinstall the plugin, and launch DD2 again."
      }
    }

    if ($RequireBattle) {
      if ($null -eq $lastState) {
        throw "Server is available, but no state frame was received. Enter a battle and try again."
      }
      $heroes = @($lastState.heroes)
      $enemies = @($lastState.enemies)
      if (-not $hasLiveBattleState) {
        throw "Server is available, but live battle state is not ready. in_battle=$($lastState.in_battle) heroes=$($heroes.Count) enemies=$($enemies.Count)."
      }
      Write-Host "Live battle state found: heroes=$($heroes.Count) enemies=$($enemies.Count) round=$($lastState.round)"
    } elseif ($hasLiveBattleState) {
      $heroes = @($lastState.heroes)
      $enemies = @($lastState.enemies)
      Write-Host "Live battle state found: heroes=$($heroes.Count) enemies=$($enemies.Count) round=$($lastState.round)"
    }

    if ($modVersion) {
      Write-Host "Mod server is available: mod_version=$modVersion"
    } else {
      Write-Host "Mod server is available."
    }
    return
  } catch {
    $lastError = $_.Exception.Message
    if ($attempt -lt [Math]::Max(1, $Retries)) {
      Write-Warning "DDRL server check attempt $attempt failed: $lastError"
      Start-Sleep -Milliseconds 500
      continue
    }
  } finally {
    if ($reader) { $reader.Dispose() }
    $client.Dispose()
  }
  }

  throw "DDRL mod server check failed at ${HostName}:$Port after $([Math]::Max(1, $Retries)) attempt(s). $lastError"
}

$repoRoot = Resolve-RepoRoot
$pythonExe = Resolve-Python -Requested $Python -RepoRoot $repoRoot
$liveScript = Join-Path $repoRoot "scripts\live_ppo.py"

if ($Mode -eq "ppo") {
  $modelPath = $Model
  if (-not [System.IO.Path]::IsPathRooted($modelPath)) {
    $modelPath = Join-Path $repoRoot $modelPath
  }
  if (-not (Test-Path $modelPath)) {
    throw "Model was not found: $modelPath"
  }
} else {
  $modelPath = $Model
}

if (-not $SkipProcessCheck) {
  Assert-GameRunning -Names $GameProcessNames
}

Test-ModServer -HostName $HostName -Port $Port -Timeout $Timeout -Retries $ServerCheckRetries -ExpectedModVersion $ExpectedModVersion -RequireBattle:(-not $AllowOutOfBattle)

$argsList = @(
  $liveScript,
  "--host", $HostName,
  "--port", "$Port",
  "--max-steps", "$MaxSteps",
  "--reset-timeout", "$ResetTimeout",
  "--action-timeout", "$ActionTimeout",
  "--enemy-turn-wait", "$EnemyTurnWait",
  "--stunned-turn-wait", "$StunnedTurnWait",
  "--step-delay", "$StepDelay",
  "--mode", $Mode
)

if ($Mode -eq "ppo") {
  $argsList += @("--model", $modelPath)
}
if ($Stochastic) {
  $argsList += "--stochastic"
}
if ($Quiet) {
  $argsList += "--quiet"
}
if ($AllowPassActions -and -not $DisablePassActions) {
  $argsList += "--allow-pass-actions"
}
if ($DisableEmergencyPass) {
  $argsList += "--disable-emergency-pass"
}
if ($AllowPolicyMoveActions) {
  $argsList += "--allow-policy-move-actions"
}

Write-Host "Starting live agent: $pythonExe $($argsList -join ' ')"
& $pythonExe @argsList
exit $LASTEXITCODE

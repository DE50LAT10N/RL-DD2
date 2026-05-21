# DDRL DD2 Mod (BepInEx Mono)

Unofficial research mod for Darkest Dungeon II (Steam, Mono). The mod
loads as a BepInEx 5 plugin, publishes combat state over localhost
NDJSON/TCP, and accepts action requests from an external client.

Not affiliated with Red Hook Studios. Do not redistribute game assets.
Single-player research use only.

## What it does

- Loads inside DD2 via `BepInEx/plugins/DdRL/DdRL.Plugin.dll`.
- Starts a local TCP JSON-line server on `127.0.0.1:8765`.
- Publishes battle `state` snapshots on turn boundaries.
- Accepts `action` requests and executes them via in-process method calls
  (`method: "hook"`), no screen clicks.
- Sends `ack`, `battle_end`, `ping/pong`, and `error` control messages.

## Directory layout

- `DdRL.sln` - solution root.
- `DdRL.Plugin/` - C# plugin project.
- `DdRL.Plugin/Config/` - runtime config parsing.
- `DdRL.Plugin/Logging/` - thin logging facade.
- `DdRL.Plugin/Ipc/` - protocol and NDJSON TCP server.
- `DdRL.Plugin/State/` - class discovery, snapshots, state reader.
- `DdRL.Plugin/Actions/` - action queue and dispatcher.
- `DdRL.Plugin/Hooks/` - Harmony hooks and class/method placeholders.
- `install.ps1` - copy build output to game plugin folder.
- `NOTES.md` - reverse engineering and hookup checklist.
- `gameinfo.json` - tested game and toolchain fingerprint.

## Build prerequisites

- .NET SDK 8.0+.
- BepInEx 5 x64 (Mono) installed into DD2 folder.
- First DD2 launch after BepInEx install (generates config and logs).

## Build

```powershell
cd mod_dd2
dotnet build DdRL.sln -c Release
```

Output:
`mod_dd2/DdRL.Plugin/bin/Release/net472/DdRL.Plugin.dll`.

## Install

```powershell
.\mod_dd2\install.ps1 -GameRoot "<DD2_GAME_ROOT>" -Build
```

Expected destination:
`<GameRoot>/BepInEx/plugins/DdRL/DdRL.Plugin.dll`.

## Verify runtime

1. Launch the game.
2. Check `BepInEx/LogOutput.log` for `DDRL` startup messages.
3. Start the guarded live runner from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_live_game.ps1" -Model ".\runs\best\best_model.zip" -MaxSteps 200
```

## Protocol

Message names intentionally mirror the DD1 mod:
`hello`, `state`, `action`, `ack`, `battle_end`, `ping`, `pong`, `error`.

The plugin adds `game: "dd2"` in `hello` and always reports
`ack.method = "hook"` on successful actions.

## Legal notes

- Single-player only.
- Do not package or distribute game binaries, assets, or proprietary data.
- Keep this repository source-only.

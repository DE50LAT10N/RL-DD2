# DD2 RE notes

1. Use Mono stack: DD2 + BepInEx 5 (`Unity Mono x64`), plugin in `BepInEx/plugins/DdRL`.
2. Enable discovery in `BepInEx/config/com.rl.ddrl.cfg`:
   - `EnableDiscoveryDump = true`
   - `EnableDataDump = false`
3. Runtime dump path: `%APPDATA%/DDRL/classes.discovered.json` and `%APPDATA%/DDRL/members.discovered.json`.
4. If runtime launch is unavailable (Steam-only launch flow), collect fallback static dump from managed assemblies:
   - `IronCrown.dll` and `Assembly-CSharp.dll` via `Mono.Cecil`.
5. Current Mono candidates in `DdRL.Plugin/Hooks/ClassPaths.cs`:
   - battle/controller type: `Assets.Code.Combat.Battle`
   - turn hook: `OnStartTurn`
   - battle end: `BattleEnd`
6. State extraction now reads from `Battle -> m_BattleTeams -> Team.Actors` and `ActorInstance` fields/properties (`DisplayedHp`, `DisplayedHpMax`, `TeamPosition`, `IsLiving`, `TokenContainer`).
7. `ResolveInstance` now falls back to `UnityEngine.Object.FindObjectOfType` for Unity objects when no singleton `Instance` exists.
8. Build/install:
   - `dotnet build "D:\RL DD\mod\DdRL.sln" -c Release`
   - `& "D:\RL DD\mod\install.ps1" -GameRoot "D:\SteamLibrary\steamapps\common\Darkest Dungeon® II" -Build`
9. Live verification targets in `BepInEx/LogOutput.log`:
   - `Patched turn method: ...`
   - `Patched battle-end method: ...`
   - no `Hook install skipped`.
10. Runtime check:
    - `powershell -ExecutionPolicy Bypass -File "D:\RL DD\scripts\run_live_game.ps1" -Model "D:\RL DD\runs\best\best_model.zip" -MaxSteps 200`

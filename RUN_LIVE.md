# Run live DD2

1. Install BepInEx 5 x64 (Mono) into Darkest Dungeon II and launch once.
2. Build plugin:
   - `powershell -ExecutionPolicy Bypass -File "D:\RL DD\mod\install.ps1" -Build`
3. Optional dump mode:
   - `powershell -ExecutionPolicy Bypass -File "D:\RL DD\mod\install.ps1" -Build -DumpData`
4. Check `BepInEx/LogOutput.log` for `DDRL plugin ready`.
5. Run trained PPO in live battle:
   - `python "D:\RL DD\scripts\live_ppo.py" --model "D:\RL DD\runs\best\best_model.zip" --max-steps 200`
   - Or use guarded launcher:
     `powershell -ExecutionPolicy Bypass -File "D:\RL DD\scripts\run_live_game.ps1" -Model "D:\RL DD\runs\best\best_model.zip" -MaxSteps 200`
6. Run local simulator checks:
   - `python scripts/validate_game_data.py`
   - `python scripts/test_engine_deterministic.py`

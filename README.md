# RL DD (DD2-only)

Project is fully DD2-focused with a flat layout:

- `env/`: DD2 data model, game data loader, engine, gym env, rewards, live bridge
- `agents/`: DD2 agents (`rule_based.py`, `ppo_agent.py`)
- `scripts/`: train/evaluate/live scripts
- `configs/`: reward config, protocol schema, data schema/overrides
- `mod/`: BepInEx IL2CPP plugin (`DdRL.Plugin`)

## Quickstart

1) Validate game data dump:

`python scripts/validate_game_data.py`

2) Run deterministic engine tests:

`python scripts/test_engine_deterministic.py`

3) Train PPO:

`python scripts/train.py`

4) Build/install plugin for live DD2:

`pwsh mod/install.ps1 -Build`

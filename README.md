# DDRL: Darkest Dungeon II Reinforcement Learning Agent

DDRL is an experimental reinforcement-learning and systems-integration project for training a tactical PPO agent against a **Darkest Dungeon II-inspired combat simulator** and optionally testing it against a locally owned copy of **Darkest Dungeon II**.

This repository is intended primarily as a portfolio/resume project demonstrating reinforcement learning, simulator design, reward shaping, curriculum training, IPC, C# plugin development, and production-oriented project hygiene. It is not affiliated with, endorsed by, sponsored by, or approved by Red Hook Studios, Valve, Steam, or the BepInEx maintainers.

The goal is to move tactical behavior into the model rather than hand-written live fallbacks: healing priority, stress management, movement discipline, target selection, tokens, DOTs, Death's Door, and action legality are represented in the simulator and reward function so the agent can learn them.


## Tech Stack

- Python 3.12
- PyTorch
- Gymnasium
- Stable-Baselines3 and `sb3-contrib` MaskablePPO
- NumPy, Pandas, Matplotlib, TensorBoard
- Pydantic and PyYAML for simulator data
- C# / .NET `netstandard2.1`
- BepInEx plugin runtime for Darkest Dungeon II
- PowerShell helper scripts for plugin install and live launch

## Repository Structure

```text
agents/                  PPO wrapper utilities
configs/                 reward weights, protocol metadata, training party overrides
env/                     simulator, data model, rewards, tactical drills, live state parser
mod/                     BepInEx plugin and optional mod-side Python diagnostics
notebooks/               training dashboard notebook
scripts/                 training, evaluation, live play, replay, promotion, readiness tools
tests/fixtures/          compact game-data fixtures used by the simulator
runs/best/               canonical best model location
runs/dd2_ppo_110m_v1/    final selected long-run model artifact
```

## Installation

You need a legally obtained local copy of Darkest Dungeon II to test live integration. Offline simulator training and evaluation do not require distributing or bundling game files.

Create and activate a virtual environment, then install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
### BepInEx Setup

The live plugin requires BepInEx to be installed in the Darkest Dungeon II game folder before running `mod/install.ps1`.

1. Download a BepInEx **Unity IL2CPP Windows x64** build from the official BepInEx sources:
   - Docs: https://docs.bepinex.dev/master/articles/user_guide/installation/unity_il2cpp.html
   - Builds: https://builds.bepinex.dev/projects/bepinex_be

2. Extract the archive into the DD2 game root, the same folder that contains `Darkest Dungeon II.exe`.

   Example:

   ```text
   D:\SteamLibrary\steamapps\common\Darkest Dungeon II\
     BepInEx\
     doorstop_config.ini
     winhttp.dll
     Darkest Dungeon II.exe
   ```

3. Start the game once and close it after the main menu loads. BepInEx should generate:

   ```text
   BepInEx\config\
   BepInEx\LogOutput.log
   ```

   The first IL2CPP launch can take longer than usual because BepInEx generates interop files.

4. If `BepInEx\LogOutput.log` exists and does not show loader errors, install the DDRL plugin with the command below.


Build and install the DD2 plugin into your game folder:

```powershell
powershell -ExecutionPolicy Bypass -File ".\mod\install.ps1" `
  -Build `
  -GameRoot "D:\SteamLibrary\steamapps\common\Darkest Dungeon II"
```

Restart the game after installing or updating the plugin.

## Training

Start a long PPO training run:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --steps 10000000 `
  --seed 123 `
  --n-envs 16 `
  --device cuda `
  --learning-rate 0.00025 `
  --lr-end-ratio 0.08 `
  --ent-coef 0.01 `
  --net-arch 512,512 `
  --n-steps 2048 `
  --batch-size 1024 `
  --n-epochs 6 `
  --gamma 0.995 `
  --max-episode-steps 120 `
  --best-model-save-path runs\best `
  --out runs\dd2_ppo_110m_v1\final_model.zip `
  --run-meta-out runs\dd2_ppo_110m_v1\run_meta.json
```

If Windows multiprocessing blocks `SubprocVecEnv`, add:

```powershell
--use-dummy-vec
```

## Evaluation

Offline holdout evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --model runs\best\best_model.zip `
  --episodes 200 `
  --seeds 17,27,37 `
  --out-json runs\eval\holdout.json
```

Pre-live readiness gate:

```powershell
.\.venv\Scripts\python.exe scripts\check_model_ready.py `
  --model runs\best\best_model.zip `
  --episodes 80 `
  --seeds 17,27,37
```

## Live Play

Live play is optional and experimental. Use it only with your own local single-player installation and only if you accept the legal and technical risks.

With DD2 running and the plugin loaded:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_live_game.ps1" `
  -Mode ppo `
  -Model ".\runs\best\best_model.zip" `
  -MaxSteps 40 `
  -LogLegalActions `
  -ActionLog ".\runs\live_action_log.jsonl"
```

The live runner uses the game's `legal_actions` as the validity boundary. Tactical preferences should be learned through simulator rewards and drills rather than live-only action overrides.

## Model Selection

The canonical best model is:

```text
runs/best/best_model.zip
```

The selected final model is:

```text
runs/dd2_ppo_110m_v1/final_model.zip
```

This final artifact is selected because `dd2_ppo_110m_v1` is the highest-version training run present in the repository and its `final_model.zip` is the largest model artifact among the available final models, which is a reasonable proxy when loss curves are not stored in source-controlled metadata.


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
runs/best/               canonical PPO model location
runs/dd2_ppo_10m_v1/     selected PPO 10M final model for comparison
runs/dd2_qrdqn_10m_v1/   selected QR-DQN model and compact eval metadata
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

Train a Masked QR-DQN comparison agent:

```powershell
.\.venv\Scripts\python.exe scripts\train_qrdqn.py `
  --steps 10000000 `
  --seed 123 `
  --n-envs 4 `
  --device cuda `
  --learning-rate 0.0001 `
  --lr-end-ratio 0.08 `
  --net-arch 384,384 `
  --n-quantiles 101 `
  --buffer-size 1000000 `
  --learning-starts 50000 `
  --batch-size 512 `
  --gamma 0.995 `
  --train-freq 4 `
  --gradient-steps 1 `
  --target-update-interval 10000 `
  --exploration-final-eps 0.05 `
  --max-episode-steps 120 `
  --best-model-save-path runs\dd2_qrdqn_10m_v1\best `
  --out runs\dd2_qrdqn_10m_v1\final_model.zip `
  --run-meta-out runs\dd2_qrdqn_10m_v1\run_meta.json
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

Compare PPO, QR-DQN, and simulator baselines on the same holdout seeds:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_compare.py `
  --ppo-model runs\best\best_model.zip `
  --qrdqn-model runs\dd2_qrdqn_10m_v1\best\best_model.zip `
  --baselines scripted,random `
  --episodes 200 `
  --seeds 17,27,37 `
  --out-json runs\eval\compare_ppo_qrdqn_dashboard.json
```

Gate QR-DQN as a meaningful PPO comparison agent:

```powershell
.\.venv\Scripts\python.exe scripts\check_qrdqn_ready.py `
  --compare-json runs\eval\compare_ppo_qrdqn_dashboard.json `
  --min-win-rate 0.35 `
  --min-ppo-win-ratio 0.70 `
  --min-scripted-win-ratio 0.50 `
  --out-json runs\eval\qrdqn_ready_dashboard.json
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

Run a trained QR-DQN comparison model live:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_live_game.ps1" `
  -Mode qrdqn `
  -Model ".\runs\dd2_qrdqn_10m_v1\best\best_model.zip" `
  -MaxSteps 40 `
  -LogLegalActions `
  -ActionLog ".\runs\live_qrdqn_action_log.jsonl"
```

The live runner uses the game's `legal_actions` as the validity boundary. Tactical preferences should be learned through simulator rewards and drills rather than live-only action overrides.

## Model Artifacts

The repository intentionally keeps only the compact artifacts needed to reproduce evaluation and live demos. Intermediate checkpoints, TensorBoard logs, smoke runs, and local live logs are ignored.

The canonical PPO model is:

```text
runs/best/best_model.zip
```

The PPO 10M final model is kept for a fuller comparison snapshot:

```text
runs/dd2_ppo_10m_v1/final_model.zip
```

The canonical QR-DQN comparison model is:

```text
runs/dd2_qrdqn_10m_v1/best/best_model.zip
```

The QR-DQN run metadata and compact evaluation outputs are kept with the model:

```text
runs/dd2_qrdqn_10m_v1/run_meta.json
runs/dd2_qrdqn_10m_v1/eval/evaluations.npz
runs/dd2_qrdqn_10m_v1/holdout_eval.json
runs/eval/compare_ppo_qrdqn_dashboard.json
runs/eval/qrdqn_ready_dashboard.json
```

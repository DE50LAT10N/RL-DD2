from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from agents.ppo_agent import PPOAgent
from env.dd_env import DarkestDungeonEnv


HOLDOUT_ENCOUNTERS = [
    "holdout_double_ghoul",
    "holdout_double_cultist",
    "holdout_mix_wave",
    "holdout_elite_pair",
    "holdout_elite_swarm",
]
PROTECTED_BEST_MODEL_DIR = (PROJECT_ROOT / "runs" / "best").resolve()
MIN_PROTECTED_BEST_STEPS = 1_000_000
PHASE1 = ["gaunt_pair", "lost_battalion_road", "road_fight"]
PHASE2 = ["gaunt_pair", "cultist_trio", "military_squad", "lost_battalion_road", "swine_pair", "road_fight"]
PHASE2B = PHASE2 + ["ghoul_solo"]
PHASE3_PAIR = PHASE2B + ["holdout_elite_pair"]
PHASE3_FULL = PHASE3_PAIR + ["holdout_elite_swarm"]
EASY_ENCOUNTERS = {"gaunt_pair", "lost_battalion_road", "road_fight"}


class CurriculumEnv(DarkestDungeonEnv):
    def __init__(self, *args, total_steps: int = 1_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = total_steps
        self.global_steps = 0
        self._current_encounter: str | None = None
        self._encounter_stats: dict[str, dict[str, int]] = {}
        self._easy_ratio = 0.2

    def _phase_encounters(self) -> list[str]:
        """Fraction of per-env steps (see note: each parallel env advances its own counter)."""
        t = max(1, int(self.total_steps))
        r = min(1.0, self.global_steps / t)
        if r < 0.22:
            return PHASE1
        if r < 0.55:
            return PHASE2
        if r < 0.72:
            return PHASE2B
        if r < 0.88:
            return PHASE3_PAIR
        return PHASE3_FULL

    def _choose_encounter(self) -> str:
        pool = self._phase_encounters()
        if not pool:
            return "road_fight"
        easy_pool = [x for x in pool if x in EASY_ENCOUNTERS]
        hard_pool = [x for x in pool if x not in EASY_ENCOUNTERS]
        if hard_pool and easy_pool and self.backend.rng.random() < self._easy_ratio:
            return self.backend.rng.choice(easy_pool)

        weighted_pool = hard_pool or pool
        weights: list[float] = []
        for encounter_id in weighted_pool:
            row = self._encounter_stats.get(encounter_id)
            if not row or row["episodes"] < 3:
                weights.append(1.0)
                continue
            win_rate = row["wins"] / max(1, row["episodes"])
            # Oversample encounters the policy still loses often.
            weights.append(max(0.1, 2.4 - 2.6 * win_rate))
        return self.backend.rng.choices(weighted_pool, weights=weights, k=1)[0]

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if "encounter_id" not in opts:
            opts["encounter_id"] = self._choose_encounter()
        self._current_encounter = str(opts["encounter_id"])
        return super().reset(seed=seed, options=opts)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        self.global_steps += 1
        if terminated or truncated:
            encounter_id = self._current_encounter or "unknown"
            row = self._encounter_stats.setdefault(encounter_id, {"episodes": 0, "wins": 0})
            row["episodes"] += 1
            row["wins"] += 1 if info.get("heroes_won") else 0
            recent = [r["wins"] / max(1, r["episodes"]) for r in self._encounter_stats.values() if r["episodes"] >= 3]
            if recent:
                avg_win = sum(recent) / len(recent)
                if avg_win < 0.45:
                    self._easy_ratio = 0.35
                elif avg_win < 0.6:
                    self._easy_ratio = 0.28
                elif avg_win < 0.75:
                    self._easy_ratio = 0.18
                else:
                    self._easy_ratio = 0.1
        return obs, reward, terminated, truncated, info


class HoldoutEvalEnv(DarkestDungeonEnv):
    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if "encounter_id" not in opts:
            opts["encounter_id"] = self.backend.rng.choice(HOLDOUT_ENCOUNTERS)
        return super().reset(seed=seed, options=opts)


class MilestoneCheckpointCallback(BaseCallback):
    def __init__(self, milestones: list[int], save_path: str = "runs/checkpoints", verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.milestones = sorted({int(x) for x in milestones if int(x) > 0})
        self.save_path = Path(save_path)
        self._saved: set[int] = set()

    def _on_step(self) -> bool:
        for milestone in self.milestones:
            if milestone in self._saved:
                continue
            if self.num_timesteps >= milestone:
                self.save_path.mkdir(parents=True, exist_ok=True)
                out = self.save_path / f"milestone_{milestone}_steps.zip"
                self.model.save(out)
                self._saved.add(milestone)
                if self.verbose:
                    print(f"saved_milestone_checkpoint={out}")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DD2 PPO in simulator")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--out", type=str, default="runs/dd2_ppo.zip")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--use-dummy-vec", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=str, default="384,384", help="Comma separated layer sizes, e.g. 384,384,256")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per env per PPO update.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument(
        "--lr-end-ratio",
        type=float,
        default=0.08,
        help="If <1, linearly decay learning rate to this fraction of --learning-rate by end of training.",
    )
    parser.add_argument("--run-meta-out", type=str, default="", help="Optional JSON path for run configuration metadata.")
    parser.add_argument("--best-model-save-path", type=str, default="runs/dev/best/", help="Directory for eval best_model.zip.")
    parser.add_argument("--eval-log-path", type=str, default="runs/eval/", help="Directory for eval metrics.")
    parser.add_argument("--checkpoint-save-path", type=str, default="runs/checkpoints/", help="Directory for periodic and milestone checkpoints.")
    parser.add_argument(
        "--eval-global-freq",
        type=int,
        default=25_000,
        help="Run holdout evaluation every N global timesteps.",
    )
    parser.add_argument(
        "--checkpoint-global-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every N global timesteps.",
    )
    parser.add_argument(
        "--milestone-checkpoints",
        type=str,
        default="100000000",
        help="Comma-separated global timesteps to save exact milestone checkpoints.",
    )
    parser.add_argument(
        "--allow-runs-best-smoke",
        action="store_true",
        help="Allow short runs to overwrite runs/best/best_model.zip. Intended only for deliberate debugging.",
    )
    return parser.parse_args()


def _make_env(seed: int, max_episode_steps: int, total_steps: int) -> Callable[[], CurriculumEnv]:
    def _factory() -> CurriculumEnv:
        return CurriculumEnv(seed=seed, max_episode_steps=max_episode_steps, total_steps=total_steps)

    return _factory


def main() -> int:
    args = parse_args()
    best_model_dir = Path(args.best_model_save_path)
    if not best_model_dir.is_absolute():
        best_model_dir = PROJECT_ROOT / best_model_dir
    best_model_dir = best_model_dir.resolve()
    if (
        best_model_dir == PROTECTED_BEST_MODEL_DIR
        and args.steps < MIN_PROTECTED_BEST_STEPS
        and not args.allow_runs_best_smoke
    ):
        raise ValueError(
            "Refusing to write a short training run to runs/best/best_model.zip. "
            f"steps={args.steps} is below {MIN_PROTECTED_BEST_STEPS}. "
            "Use a run-local --best-model-save-path for smoke tests, or pass "
            "--allow-runs-best-smoke if this overwrite is intentional."
        )
    if args.device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = args.device
    if selected_device == "cuda" and not torch.cuda.is_available():
        print("requested cuda but unavailable; falling back to cpu")
        selected_device = "cpu"
    print(f"training_device={selected_device}")

    vec_cls = DummyVecEnv if args.use_dummy_vec else SubprocVecEnv
    per_env_total_steps = max(1, args.steps // max(1, args.n_envs))
    env_fns = [_make_env(args.seed + i, args.max_episode_steps, per_env_total_steps) for i in range(args.n_envs)]
    env = vec_cls(env_fns)
    eval_env = DummyVecEnv([lambda: Monitor(HoldoutEvalEnv(seed=args.seed + 997, max_episode_steps=args.max_episode_steps))])
    checkpoint_freq = max(1, args.checkpoint_global_freq // max(1, args.n_envs))
    eval_freq = max(1, args.eval_global_freq // max(1, args.n_envs))
    milestone_checkpoints = [
        int(x.strip().replace("_", ""))
        for x in str(args.milestone_checkpoints).split(",")
        if x.strip()
    ]
    callbacks = CallbackList([
        CheckpointCallback(save_freq=checkpoint_freq, save_path=args.checkpoint_save_path),
        MilestoneCheckpointCallback(milestone_checkpoints, save_path=args.checkpoint_save_path, verbose=1),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=args.best_model_save_path,
            log_path=args.eval_log_path,
            eval_freq=eval_freq,
            n_eval_episodes=20,
            deterministic=True,
        ),
    ])
    net_arch = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]
    lr = float(args.learning_rate)
    if 0 < args.lr_end_ratio < 1.0:

        def learning_rate_schedule(progress_remaining: float) -> float:
            return lr * (args.lr_end_ratio + (1.0 - args.lr_end_ratio) * progress_remaining)

        lr_arg: float | Callable[[float], float] = learning_rate_schedule
    else:
        lr_arg = lr
    if args.resume:
        model = MaskablePPO.load(args.resume, device=selected_device)
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError(
                "Resume model is incompatible with the updated simulator spaces. "
                f"model_obs={model.observation_space} env_obs={env.observation_space}; "
                f"model_action={model.action_space} env_action={env.action_space}. "
                "Start a fresh training run for pass/move/remnant support."
            )
        model.set_env(env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_arg,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            tensorboard_log="runs/tb/",
            policy_kwargs={"net_arch": net_arch},
            device=selected_device,
        )
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    Path("runs").mkdir(exist_ok=True)
    PPOAgent(model).save(args.out)
    if args.run_meta_out:
        meta = {
            "device": selected_device,
            "seed": args.seed,
            "steps": args.steps,
            "n_envs": args.n_envs,
            "learning_rate": args.learning_rate,
            "lr_end_ratio": args.lr_end_ratio,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "net_arch": net_arch,
            "resume": args.resume,
            "out": args.out,
            "best_model_save_path": args.best_model_save_path,
            "eval_log_path": args.eval_log_path,
            "checkpoint_save_path": args.checkpoint_save_path,
            "eval_global_freq": args.eval_global_freq,
            "checkpoint_global_freq": args.checkpoint_global_freq,
            "milestone_checkpoints": milestone_checkpoints,
            "curriculum_total_steps_per_env": per_env_total_steps,
        }
        meta_path = Path(args.run_meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote_run_meta={meta_path}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

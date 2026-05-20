# Main MaskablePPO training entry point.
# Builds curriculum environments, tactical drills, evaluation callbacks, checkpoints, and final model output.
# Depends on sb3-contrib MaskablePPO and the Gymnasium DD2 simulator.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, sync_envs_normalization
import torch

from agents.ppo_agent import PPOAgent
from env.curriculum import CurriculumEnv, HoldoutEvalEnv


PROTECTED_BEST_MODEL_DIR = (PROJECT_ROOT / "runs" / "best").resolve()
MIN_PROTECTED_BEST_STEPS = 1_000_000


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


class SmoothedMaskableEvalCallback(MaskableEvalCallback):
    def __init__(
        self,
        *args,
        best_min_timesteps: int = 0,
        best_window_evals: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.best_min_timesteps = max(0, int(best_min_timesteps))
        self.best_window_evals = max(1, int(best_window_evals))
        self._mean_reward_history: list[float] = []

    def _best_score_ready(self) -> bool:
        return self.num_timesteps >= self.best_min_timesteps and len(self._mean_reward_history) >= self.best_window_evals

    def _best_score(self) -> float:
        window = self._mean_reward_history[-self.best_window_evals :]
        return float(np.mean(window))

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            self._is_success_buffer = []
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,  # type: ignore[arg-type]
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                use_masking=self.use_masking,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)
            self._mean_reward_history.append(float(mean_reward))
            score_ready = self._best_score_ready()
            best_score = self._best_score() if score_ready else float("-inf")

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                if score_ready:
                    print(
                        f"Best score candidate: smoothed_reward={best_score:.2f} "
                        f"window_evals={self.best_window_evals} min_timesteps={self.best_min_timesteps}"
                    )
                else:
                    print(
                        "Best score candidate: waiting "
                        f"evals={len(self._mean_reward_history)}/{self.best_window_evals} "
                        f"timesteps={self.num_timesteps}/{self.best_min_timesteps}"
                    )

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            if score_ready:
                self.logger.record("eval/smoothed_best_score", best_score)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if score_ready and best_score > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best smoothed reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = best_score
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DD2 PPO in simulator")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--out", type=str, default="runs/dd2_ppo.zip")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--use-dummy-vec", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--net-arch", type=str, default="384,384", help="Comma separated layer sizes, e.g. 384,384,256")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per env per PPO update.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument(
        "--lr-end-ratio",
        type=float,
        default=0.04,
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
        "--eval-episodes",
        type=int,
        default=80,
        help="Holdout episodes per eval callback run.",
    )
    parser.add_argument(
        "--best-min-global-steps",
        type=int,
        default=1_000_000,
        help="Do not overwrite best_model.zip before this many global timesteps.",
    )
    parser.add_argument(
        "--best-reward-window-evals",
        type=int,
        default=3,
        help="Use the mean reward over this many latest evals when deciding best_model.zip.",
    )
    parser.add_argument(
        "--drill-ratio",
        type=float,
        default=0.28,
        help="Fraction of training resets mutated into tactical drill states.",
    )
    parser.add_argument(
        "--critical-heal-drill-ratio",
        type=float,
        default=0.18,
        help="Fraction of training resets that start with Plague Doctor acting and an ally at critical HP.",
    )
    parser.add_argument(
        "--critical-heal-drill-max-hp-ratio",
        type=float,
        default=0.25,
        help="Maximum victim HP ratio for critical-heal drill resets.",
    )
    parser.add_argument(
        "--eval-critical-heal-drill-ratio",
        type=float,
        default=0.25,
        help="Fraction of holdout eval resets that include a critical-HP Plague Doctor heal decision.",
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


def _make_env(
    seed: int,
    max_episode_steps: int,
    total_steps: int,
    critical_heal_drill_ratio: float,
    critical_heal_drill_max_hp_ratio: float,
    drill_ratio: float,
) -> Callable[[], CurriculumEnv]:
    def _factory() -> CurriculumEnv:
        return CurriculumEnv(
            seed=seed,
            max_episode_steps=max_episode_steps,
            total_steps=total_steps,
            drill_ratio=drill_ratio,
            critical_heal_drill_ratio=critical_heal_drill_ratio,
            critical_heal_drill_max_hp_ratio=critical_heal_drill_max_hp_ratio,
        )

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
    env_fns = [
        _make_env(
            args.seed + i,
            args.max_episode_steps,
            per_env_total_steps,
            args.critical_heal_drill_ratio,
            args.critical_heal_drill_max_hp_ratio,
            args.drill_ratio,
        )
        for i in range(args.n_envs)
    ]
    env = vec_cls(env_fns)
    eval_env = DummyVecEnv([
        lambda: Monitor(
            HoldoutEvalEnv(
                seed=args.seed + 997,
                max_episode_steps=args.max_episode_steps,
                critical_heal_drill_ratio=args.eval_critical_heal_drill_ratio,
                critical_heal_drill_max_hp_ratio=args.critical_heal_drill_max_hp_ratio,
            )
        )
    ])
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
        SmoothedMaskableEvalCallback(
            eval_env,
            best_model_save_path=args.best_model_save_path,
            log_path=args.eval_log_path,
            eval_freq=eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            best_min_timesteps=args.best_min_global_steps,
            best_window_evals=args.best_reward_window_evals,
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
            "eval_episodes": args.eval_episodes,
            "best_min_global_steps": args.best_min_global_steps,
            "best_reward_window_evals": args.best_reward_window_evals,
            "drill_ratio": args.drill_ratio,
            "critical_heal_drill_ratio": args.critical_heal_drill_ratio,
            "critical_heal_drill_max_hp_ratio": args.critical_heal_drill_max_hp_ratio,
            "eval_critical_heal_drill_ratio": args.eval_critical_heal_drill_ratio,
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

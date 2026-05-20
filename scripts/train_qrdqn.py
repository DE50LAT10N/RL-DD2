# Main MaskedQRDQN training entry point.
# Uses the same curriculum and holdout environments as PPO for fair comparison.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from agents.masked_qrdqn import MaskedQRDQN
from agents.qrdqn_agent import QRDQNAgent
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


class SmoothedMaskedEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: HoldoutEvalEnv,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        log_path: str,
        best_min_timesteps: int = 0,
        best_window_evals: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.best_model_save_path = Path(best_model_save_path)
        self.log_path = Path(log_path) if log_path else None
        self.best_min_timesteps = max(0, int(best_min_timesteps))
        self.best_window_evals = max(1, int(best_window_evals))
        self.best_mean_reward = float("-inf")
        self._last_eval_step = 0
        self._mean_reward_history: list[float] = []
        self.evaluations_timesteps: list[int] = []
        self.evaluations_results: list[list[float]] = []
        self.evaluations_length: list[list[int]] = []
        self.evaluations_successes: list[list[bool]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        rewards, lengths, successes = evaluate_masked_qrdqn(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
        )
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_ep_length = float(np.mean(lengths))
        success_rate = float(np.mean(successes))
        self._mean_reward_history.append(mean_reward)
        score_ready = (
            self.num_timesteps >= self.best_min_timesteps
            and len(self._mean_reward_history) >= self.best_window_evals
        )
        window = self._mean_reward_history[-self.best_window_evals :]
        smoothed_score = float(np.mean(window)) if score_ready else float("-inf")

        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append([float(x) for x in rewards])
        self.evaluations_length.append([int(x) for x in lengths])
        self.evaluations_successes.append([bool(x) for x in successes])
        self._save_eval_log()

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f}")
            print(f"Success rate: {100 * success_rate:.2f}%")
            if score_ready:
                print(f"Best score candidate: smoothed_reward={smoothed_score:.2f}")

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.record("eval/success_rate", success_rate)
        if score_ready:
            self.logger.record("eval/smoothed_best_score", smoothed_score)
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        if score_ready and smoothed_score > self.best_mean_reward:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_save_path / "best_model")
            self.best_mean_reward = smoothed_score
            if self.verbose > 0:
                print("New best smoothed reward!")
        return True

    def _save_eval_log(self) -> None:
        if self.log_path is None:
            return
        self.log_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.log_path / "evaluations.npz",
            timesteps=self.evaluations_timesteps,
            results=self.evaluations_results,
            ep_lengths=self.evaluations_length,
            successes=self.evaluations_successes,
        )


def evaluate_masked_qrdqn(
    model: MaskedQRDQN,
    env: HoldoutEvalEnv,
    *,
    n_eval_episodes: int,
) -> tuple[list[float], list[int], list[bool]]:
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=_env_action_masks(env))
            obs, reward, terminated, truncated, info = env.step(int(action[0]))
            ep_reward += float(reward)
            ep_length += 1
            done = bool(terminated or truncated)
        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(bool(info.get("heroes_won")))
    return rewards, lengths, successes


def _env_action_masks(env) -> np.ndarray:
    if hasattr(env, "action_masks"):
        return env.action_masks()
    if hasattr(env, "get_wrapper_attr"):
        return env.get_wrapper_attr("action_masks")()
    return env.unwrapped.action_masks()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DD2 MaskedQRDQN in simulator")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--out", type=str, default="runs/dd2_qrdqn.zip")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--use-dummy-vec", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lr-end-ratio", type=float, default=0.08)
    parser.add_argument("--net-arch", type=str, default="384,384")
    parser.add_argument("--n-quantiles", type=int, default=101)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.18)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--run-meta-out", type=str, default="")
    parser.add_argument("--best-model-save-path", type=str, default="runs/dev/qrdqn_best/")
    parser.add_argument("--eval-log-path", type=str, default="runs/eval_qrdqn/")
    parser.add_argument("--checkpoint-save-path", type=str, default="runs/qrdqn_checkpoints/")
    parser.add_argument("--eval-global-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--best-min-global-steps", type=int, default=1_000_000)
    parser.add_argument("--best-reward-window-evals", type=int, default=3)
    parser.add_argument("--drill-ratio", type=float, default=0.28)
    parser.add_argument("--critical-heal-drill-ratio", type=float, default=0.18)
    parser.add_argument("--critical-heal-drill-max-hp-ratio", type=float, default=0.25)
    parser.add_argument("--eval-critical-heal-drill-ratio", type=float, default=0.25)
    parser.add_argument("--checkpoint-global-freq", type=int, default=50_000)
    parser.add_argument("--milestone-checkpoints", type=str, default="100000000")
    parser.add_argument("--allow-runs-best-smoke", action="store_true")
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


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but unavailable; falling back to cpu")
        return "cpu"
    return requested


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

    selected_device = _select_device(args.device)
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
    eval_env = Monitor(
        HoldoutEvalEnv(
            seed=args.seed + 997,
            max_episode_steps=args.max_episode_steps,
            critical_heal_drill_ratio=args.eval_critical_heal_drill_ratio,
            critical_heal_drill_max_hp_ratio=args.critical_heal_drill_max_hp_ratio,
        )
    )

    checkpoint_freq = max(1, args.checkpoint_global_freq // max(1, args.n_envs))
    milestone_checkpoints = [
        int(x.strip().replace("_", ""))
        for x in str(args.milestone_checkpoints).split(",")
        if x.strip()
    ]
    callbacks = CallbackList([
        CheckpointCallback(save_freq=checkpoint_freq, save_path=args.checkpoint_save_path),
        MilestoneCheckpointCallback(milestone_checkpoints, save_path=args.checkpoint_save_path, verbose=1),
        SmoothedMaskedEvalCallback(
            eval_env,
            best_model_save_path=args.best_model_save_path,
            log_path=args.eval_log_path,
            eval_freq=args.eval_global_freq,
            n_eval_episodes=args.eval_episodes,
            best_min_timesteps=args.best_min_global_steps,
            best_window_evals=args.best_reward_window_evals,
            verbose=1,
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
        model = MaskedQRDQN.load(args.resume, env=env, device=selected_device)
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError(
                "Resume model is incompatible with the current simulator spaces. "
                f"model_obs={model.observation_space} env_obs={env.observation_space}; "
                f"model_action={model.action_space} env_action={env.action_space}."
            )
        model.set_env(env)
    else:
        model = MaskedQRDQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_arg,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log="runs/tb/",
            policy_kwargs={"net_arch": net_arch, "n_quantiles": args.n_quantiles},
            device=selected_device,
            seed=args.seed,
        )

    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    QRDQNAgent(model).save(args.out)
    if args.run_meta_out:
        meta = {
            "algorithm": "MaskedQRDQN",
            "device": selected_device,
            "seed": args.seed,
            "steps": args.steps,
            "n_envs": args.n_envs,
            "learning_rate": args.learning_rate,
            "lr_end_ratio": args.lr_end_ratio,
            "net_arch": net_arch,
            "n_quantiles": args.n_quantiles,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "target_update_interval": args.target_update_interval,
            "exploration_fraction": args.exploration_fraction,
            "exploration_initial_eps": args.exploration_initial_eps,
            "exploration_final_eps": args.exploration_final_eps,
            "max_grad_norm": args.max_grad_norm,
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

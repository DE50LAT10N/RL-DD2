# Staged training orchestrator.
# Runs scripts/train.py across increasing milestone budgets while resuming from prior stage outputs.
# Useful for long training runs with checkpointed progression.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run staged MaskablePPO training milestones.")
    p.add_argument("--milestones", type=str, default="1000000,3000000,5000000,100000000")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--use-dummy-vec", action="store_true")
    p.add_argument("--learning-rate", type=float, default=1.5e-4)
    p.add_argument("--lr-end-ratio", type=float, default=0.04)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--net-arch", type=str, default="384,384")
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--max-episode-steps", type=int, default=180)
    p.add_argument("--checkpoint-global-freq", type=int, default=50_000)
    p.add_argument("--checkpoint-save-path", type=str, default="runs/checkpoints/")
    p.add_argument("--milestone-checkpoints", type=str, default="100000000")
    p.add_argument("--eval-global-freq", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--best-min-global-steps", type=int, default=1_000_000)
    p.add_argument("--best-reward-window-evals", type=int, default=3)
    p.add_argument("--drill-ratio", type=float, default=0.28)
    p.add_argument("--critical-heal-drill-ratio", type=float, default=0.18)
    p.add_argument("--critical-heal-drill-max-hp-ratio", type=float, default=0.25)
    p.add_argument("--eval-critical-heal-drill-ratio", type=float, default=0.25)
    p.add_argument("--best-model-save-path", type=str, default="runs/dev/staged_best/")
    p.add_argument("--eval-log-path", type=str, default="runs/eval/")
    p.add_argument("--out-prefix", type=str, default="runs/dd2_ppo_stage")
    p.add_argument(
        "--allow-runs-best-smoke",
        action="store_true",
        help="Forward the protected runs/best smoke override to scripts/train.py.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    milestones = sorted({int(x.strip().replace("_", "")) for x in args.milestones.split(",") if x.strip()})
    if not milestones:
        raise ValueError("No milestones specified.")
    if milestones[0] <= 0:
        raise ValueError("Milestones must be positive.")

    python = sys.executable
    train_py = PROJECT_ROOT / "scripts" / "train.py"
    resume_path: Path | None = None

    for idx, stage_steps in enumerate(milestones, start=1):
        out_model = PROJECT_ROOT / f"{args.out_prefix}_{stage_steps}.zip"
        out_meta = PROJECT_ROOT / f"{args.out_prefix}_{stage_steps}_meta.json"
        cmd = [
            python,
            str(train_py),
            "--steps",
            str(stage_steps),
            "--seed",
            str(args.seed),
            "--n-envs",
            str(args.n_envs),
            "--device",
            args.device,
            "--learning-rate",
            str(args.learning_rate),
            "--lr-end-ratio",
            str(args.lr_end_ratio),
            "--ent-coef",
            str(args.ent_coef),
            "--net-arch",
            args.net_arch,
            "--n-steps",
            str(args.n_steps),
            "--batch-size",
            str(args.batch_size),
            "--n-epochs",
            str(args.n_epochs),
            "--gamma",
            str(args.gamma),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--checkpoint-global-freq",
            str(args.checkpoint_global_freq),
            "--checkpoint-save-path",
            args.checkpoint_save_path,
            "--eval-global-freq",
            str(args.eval_global_freq),
            "--eval-episodes",
            str(args.eval_episodes),
            "--best-min-global-steps",
            str(args.best_min_global_steps),
            "--best-reward-window-evals",
            str(args.best_reward_window_evals),
            "--drill-ratio",
            str(args.drill_ratio),
            "--critical-heal-drill-ratio",
            str(args.critical_heal_drill_ratio),
            "--critical-heal-drill-max-hp-ratio",
            str(args.critical_heal_drill_max_hp_ratio),
            "--eval-critical-heal-drill-ratio",
            str(args.eval_critical_heal_drill_ratio),
            "--milestone-checkpoints",
            args.milestone_checkpoints,
            "--best-model-save-path",
            args.best_model_save_path,
            "--eval-log-path",
            args.eval_log_path,
            "--out",
            str(out_model),
            "--run-meta-out",
            str(out_meta),
        ]
        if args.use_dummy_vec:
            cmd.append("--use-dummy-vec")
        if args.allow_runs_best_smoke:
            cmd.append("--allow-runs-best-smoke")
        if resume_path and resume_path.exists():
            cmd.extend(["--resume", str(resume_path)])

        print(f"[stage {idx}/{len(milestones)}] steps={stage_steps} resume={resume_path}")
        subprocess.run(cmd, check=True)
        resume_path = out_model

    print(f"completed_stages={','.join(str(x) for x in milestones)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

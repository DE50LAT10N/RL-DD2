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
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--lr-end-ratio", type=float, default=0.08)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--net-arch", type=str, default="384,384")
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--max-episode-steps", type=int, default=120)
    p.add_argument("--checkpoint-global-freq", type=int, default=50_000)
    p.add_argument("--checkpoint-save-path", type=str, default="runs/checkpoints/")
    p.add_argument("--milestone-checkpoints", type=str, default="100000000")
    p.add_argument("--eval-global-freq", type=int, default=25_000)
    p.add_argument("--best-model-save-path", type=str, default="runs/best/")
    p.add_argument("--eval-log-path", type=str, default="runs/eval/")
    p.add_argument("--out-prefix", type=str, default="runs/dd2_ppo_stage")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    milestones = sorted({int(x.strip()) for x in args.milestones.split(",") if x.strip()})
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
        if resume_path and resume_path.exists():
            cmd.extend(["--resume", str(resume_path)])

        print(f"[stage {idx}/{len(milestones)}] steps={stage_steps} resume={resume_path}")
        subprocess.run(cmd, check=True)
        resume_path = out_model

    print(f"completed_stages={','.join(str(x) for x in milestones)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Model promotion helper.
# Copies a selected trained model into the canonical runs/best location with metadata-friendly checks.
# Intended for controlled upgrades after evaluation.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote a candidate model only if holdout evaluation improves.")
    p.add_argument("--candidate", required=True, help="Candidate model zip to evaluate and possibly promote.")
    p.add_argument("--best", default="runs/best/best_model.zip", help="Current best model zip.")
    p.add_argument("--episodes", type=int, default=200, help="Evaluation episodes per seed.")
    p.add_argument("--seeds", default="17,27,37", help="Comma-separated eval seeds.")
    p.add_argument("--candidate-json", default="", help="Path for candidate evaluation JSON.")
    p.add_argument("--best-json", default="", help="Path for current-best evaluation JSON.")
    p.add_argument("--decision-json", default="", help="Path for promotion decision JSON.")
    p.add_argument(
        "--min-win-rate-delta",
        type=float,
        default=0.0,
        help="Required candidate win-rate improvement over current best.",
    )
    p.add_argument(
        "--min-mean-reward-delta",
        type=float,
        default=0.0,
        help="Tie-breaker reward delta when win rates are equal after min-win-rate-delta.",
    )
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _run_eval(model_path: Path, episodes: int, seeds: str, out_json: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--model",
        str(model_path),
        "--episodes",
        str(episodes),
        "--seeds",
        seeds,
        "--out-json",
        str(out_json),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _summary(metrics: dict[str, Any]) -> dict[str, float]:
    summary = metrics.get("summary") or {}
    return {
        "win_rate": float(summary.get("win_rate") or 0.0),
        "mean_reward": float(summary.get("mean_reward") or 0.0),
        "survival_rate": float(summary.get("survival_rate") or 0.0),
        "mean_steps": float(summary.get("mean_steps") or 0.0),
    }


def _is_better(candidate: dict[str, float], best: dict[str, float], min_win_delta: float, min_reward_delta: float) -> bool:
    win_delta = candidate["win_rate"] - best["win_rate"]
    if win_delta > min_win_delta:
        return True
    if win_delta < -min_win_delta:
        return False

    reward_delta = candidate["mean_reward"] - best["mean_reward"]
    if reward_delta > min_reward_delta:
        return True
    if reward_delta < -min_reward_delta:
        return False

    survival_delta = candidate["survival_rate"] - best["survival_rate"]
    if survival_delta != 0:
        return survival_delta > 0

    # Faster wins are preferable only after the main quality metrics tie.
    return candidate["mean_steps"] < best["mean_steps"]


def _promote(candidate_path: Path, best_path: Path) -> None:
    best_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = best_path.with_name(best_path.name + ".tmp")
    shutil.copy2(candidate_path, tmp_path)
    os.replace(tmp_path, best_path)


def main() -> int:
    args = parse_args()
    candidate_path = _resolve(args.candidate)
    best_path = _resolve(args.best)

    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate model not found: {candidate_path}")
    if candidate_path == best_path:
        raise ValueError("Promotion gate needs a run-local candidate; candidate and best point to the same file.")

    candidate_json = _resolve(args.candidate_json or str(candidate_path.with_name("candidate_eval.json")))
    best_json = _resolve(args.best_json or str(candidate_path.with_name("current_best_eval.json")))
    decision_json = _resolve(args.decision_json or str(candidate_path.with_name("promotion_decision.json")))
    candidate_json.parent.mkdir(parents=True, exist_ok=True)
    best_json.parent.mkdir(parents=True, exist_ok=True)
    decision_json.parent.mkdir(parents=True, exist_ok=True)

    candidate_metrics = _run_eval(candidate_path, args.episodes, args.seeds, candidate_json)
    candidate_summary = _summary(candidate_metrics)
    best_exists = best_path.is_file()
    if best_exists:
        best_metrics = _run_eval(best_path, args.episodes, args.seeds, best_json)
        best_summary = _summary(best_metrics)
        promoted = _is_better(
            candidate_summary,
            best_summary,
            min_win_delta=args.min_win_rate_delta,
            min_reward_delta=args.min_mean_reward_delta,
        )
    else:
        best_metrics = None
        best_summary = None
        promoted = True

    if promoted:
        _promote(candidate_path, best_path)

    decision = {
        "promoted": promoted,
        "primary_metric": "win_rate",
        "tie_breakers": ["mean_reward", "survival_rate", "mean_steps"],
        "candidate": str(candidate_path),
        "best": str(best_path),
        "candidate_summary": candidate_summary,
        "previous_best_summary": best_summary,
        "episodes_per_seed": args.episodes,
        "seeds": [int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        "min_win_rate_delta": args.min_win_rate_delta,
        "min_mean_reward_delta": args.min_mean_reward_delta,
        "candidate_json": str(candidate_json),
        "best_json": str(best_json) if best_metrics is not None else "",
    }
    decision_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"promotion_decision={json.dumps(decision, ensure_ascii=False, sort_keys=True)}")
    print(f"wrote_promotion_decision={decision_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

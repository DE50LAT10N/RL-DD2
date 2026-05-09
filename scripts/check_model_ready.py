# Pre-live model readiness gate.
# Runs simulator eval, optional live-log replay, and sim-vs-live legality checks before playing in DD2.
# Depends on evaluate.py, evaluate_live_log.py, and compare_sim_live.py.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the pre-live benchmark gate for a PPO model.")
    p.add_argument("--model", default="runs/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seeds", default="17,27,37")
    p.add_argument("--live-log", default="runs/live_action_log.jsonl")
    p.add_argument("--out-dir", default="runs/prelive")
    p.add_argument("--min-win-rate", type=float, default=0.0)
    p.add_argument("--min-composite-score", type=float, default=-1.0e9)
    p.add_argument("--max-critical-heal-miss-rate", type=float, default=0.35)
    p.add_argument("--max-bad-pass-rate", type=float, default=0.08)
    p.add_argument("--max-only-live-rate", type=float, default=0.35)
    p.add_argument("--skip-live-log", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return int(proc.returncode), proc.stdout


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _failures(args: argparse.Namespace, eval_data: dict[str, Any], live_data: dict[str, Any], diff_data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    eval_summary = eval_data.get("summary") or {}
    if float(eval_summary.get("win_rate", 0.0)) < args.min_win_rate:
        failures.append(f"win_rate<{args.min_win_rate}")
    if float(eval_summary.get("composite_score", -1.0e9)) < args.min_composite_score:
        failures.append(f"composite_score<{args.min_composite_score}")
    eval_tactical = eval_data.get("tactical_metrics") or {}
    if float(eval_tactical.get("critical_heal_miss_rate", 0.0)) > args.max_critical_heal_miss_rate:
        failures.append(f"eval_critical_heal_miss_rate>{args.max_critical_heal_miss_rate}")
    if float(eval_tactical.get("bad_pass_rate", 0.0)) > args.max_bad_pass_rate:
        failures.append(f"eval_bad_pass_rate>{args.max_bad_pass_rate}")

    if live_data:
        live_summary = live_data.get("summary") or {}
        if float(live_summary.get("critical_heal_miss_rate", 0.0)) > args.max_critical_heal_miss_rate:
            failures.append(f"critical_heal_miss_rate>{args.max_critical_heal_miss_rate}")
        if float(live_summary.get("bad_pass_rate", 0.0)) > args.max_bad_pass_rate:
            failures.append(f"bad_pass_rate>{args.max_bad_pass_rate}")

    if diff_data:
        diff_summary = diff_data.get("summary") or {}
        if float(diff_summary.get("only_live_rate", 0.0)) > args.max_only_live_rate:
            failures.append(f"only_live_rate>{args.max_only_live_rate}")
    return failures


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_json = out_dir / "eval.json"
    live_json = out_dir / "live_replay.json"
    diff_json = out_dir / "sim_live_diff.json"

    commands = [
        [
            sys.executable,
            "scripts/evaluate.py",
            "--model",
            args.model,
            "--episodes",
            str(args.episodes),
            "--seeds",
            args.seeds,
            "--out-json",
            str(eval_json),
        ]
    ]
    live_log = Path(args.live_log)
    use_live_log = live_log.is_file() and not args.skip_live_log
    if use_live_log:
        commands.extend(
            [
                [
                    sys.executable,
                    "scripts/evaluate_live_log.py",
                    "--model",
                    args.model,
                    "--log",
                    str(live_log),
                    "--out-json",
                    str(live_json),
                    "--fail-critical-heal-miss-rate",
                    "1.1",
                    "--fail-bad-pass-rate",
                    "1.1",
                ],
                [
                    sys.executable,
                    "scripts/compare_sim_live.py",
                    "--log",
                    str(live_log),
                    "--out-json",
                    str(diff_json),
                ],
            ]
        )

    command_results: list[dict[str, Any]] = []
    for cmd in commands:
        print("running=" + " ".join(cmd))
        code, output = _run(cmd)
        print(output.rstrip())
        command_results.append({"command": cmd, "returncode": code})
        if code != 0:
            summary = {"ready": False, "reason": "command_failed", "command": cmd, "returncode": code}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return code

    eval_data = _load_json(eval_json)
    live_data = _load_json(live_json) if use_live_log else {}
    diff_data = _load_json(diff_json) if use_live_log else {}
    failures = _failures(args, eval_data, live_data, diff_data)
    summary = {
        "ready": not failures,
        "failures": failures,
        "model": args.model,
        "eval": eval_data.get("summary", {}),
        "tactical": eval_data.get("tactical_metrics", {}),
        "live_replay": live_data.get("summary", {}) if live_data else "skipped",
        "sim_live_diff": diff_data.get("summary", {}) if diff_data else "skipped",
        "artifacts": {
            "eval": str(eval_json),
            "live_replay": str(live_json) if use_live_log else "",
            "sim_live_diff": str(diff_json) if use_live_log else "",
        },
        "commands": command_results,
    }
    (out_dir / "ready_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 10


if __name__ == "__main__":
    raise SystemExit(main())

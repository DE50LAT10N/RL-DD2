# Readiness gate for QR-DQN as a meaningful PPO comparison agent.
# Consumes scripts/evaluate_compare.py JSON and applies acceptance thresholds.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check QR-DQN readiness from an evaluate_compare.py JSON report.")
    p.add_argument("--compare-json", required=True, help="JSON produced by scripts/evaluate_compare.py.")
    p.add_argument("--agent", default="qrdqn", help="Agent key to gate.")
    p.add_argument("--ppo-agent", default="ppo", help="Optional PPO reference key.")
    p.add_argument("--scripted-agent", default="scripted", help="Optional scripted baseline key.")
    p.add_argument("--random-agent", default="random", help="Optional random baseline key.")
    p.add_argument("--out-json", default="", help="Optional path for readiness summary JSON.")
    p.add_argument("--min-win-rate", type=float, default=0.35)
    p.add_argument("--min-composite-score", type=float, default=0.0)
    p.add_argument("--min-survival-rate", type=float, default=0.60)
    p.add_argument("--max-bad-pass-rate", type=float, default=0.08)
    p.add_argument("--max-wasted-heal-rate", type=float, default=0.08)
    p.add_argument("--max-critical-heal-miss-rate", type=float, default=0.35)
    p.add_argument("--min-kill-confirm-rate", type=float, default=0.25)
    p.add_argument("--min-random-win-margin", type=float, default=0.10)
    p.add_argument("--min-random-composite-margin", type=float, default=5.0)
    p.add_argument("--min-scripted-win-ratio", type=float, default=0.50)
    p.add_argument("--min-ppo-win-ratio", type=float, default=0.70)
    p.add_argument("--min-ppo-composite-ratio", type=float, default=0.70)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_path = Path(args.compare_json)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    agents = data.get("agents") or {}
    if args.agent not in agents:
        raise KeyError(f"Agent '{args.agent}' not found in {report_path}")

    target = agents[args.agent]
    target_summary = target.get("summary") or {}
    target_tactical = target.get("tactical_metrics") or {}
    failures = _threshold_failures(args, target_summary, target_tactical)
    comparisons = {}

    random_data = agents.get(args.random_agent)
    if random_data:
        random_summary = random_data.get("summary") or {}
        win_margin = _num(target_summary, "win_rate") - _num(random_summary, "win_rate")
        composite_margin = _num(target_summary, "composite_score") - _num(random_summary, "composite_score")
        comparisons["random"] = {
            "win_margin": win_margin,
            "composite_margin": composite_margin,
            "reference_agent": args.random_agent,
        }
        if win_margin < args.min_random_win_margin:
            failures.append(f"random_win_margin<{args.min_random_win_margin}")
        if composite_margin < args.min_random_composite_margin:
            failures.append(f"random_composite_margin<{args.min_random_composite_margin}")

    scripted_data = agents.get(args.scripted_agent)
    if scripted_data:
        scripted_summary = scripted_data.get("summary") or {}
        scripted_win = _num(scripted_summary, "win_rate")
        scripted_ratio = _safe_ratio(_num(target_summary, "win_rate"), scripted_win)
        comparisons["scripted"] = {
            "win_ratio": scripted_ratio,
            "reference_win_rate": scripted_win,
            "reference_agent": args.scripted_agent,
        }
        if scripted_win > 0 and scripted_ratio < args.min_scripted_win_ratio:
            failures.append(f"scripted_win_ratio<{args.min_scripted_win_ratio}")

    ppo_data = agents.get(args.ppo_agent)
    if ppo_data:
        ppo_summary = ppo_data.get("summary") or {}
        ppo_win_ratio = _safe_ratio(_num(target_summary, "win_rate"), _num(ppo_summary, "win_rate"))
        ppo_composite_ratio = _safe_ratio(
            _num(target_summary, "composite_score"),
            _num(ppo_summary, "composite_score"),
        )
        comparisons["ppo"] = {
            "win_ratio": ppo_win_ratio,
            "composite_ratio": ppo_composite_ratio,
            "reference_agent": args.ppo_agent,
        }
        if _num(ppo_summary, "win_rate") > 0 and ppo_win_ratio < args.min_ppo_win_ratio:
            failures.append(f"ppo_win_ratio<{args.min_ppo_win_ratio}")
        if _num(ppo_summary, "composite_score") > 0 and ppo_composite_ratio < args.min_ppo_composite_ratio:
            failures.append(f"ppo_composite_ratio<{args.min_ppo_composite_ratio}")

    summary = {
        "ready": not failures,
        "agent": args.agent,
        "compare_json": str(report_path),
        "failures": failures,
        "target": {
            "summary": target_summary,
            "tactical_metrics": target_tactical,
        },
        "comparisons": comparisons,
        "thresholds": {
            "min_win_rate": args.min_win_rate,
            "min_composite_score": args.min_composite_score,
            "min_survival_rate": args.min_survival_rate,
            "max_bad_pass_rate": args.max_bad_pass_rate,
            "max_wasted_heal_rate": args.max_wasted_heal_rate,
            "max_critical_heal_miss_rate": args.max_critical_heal_miss_rate,
            "min_kill_confirm_rate": args.min_kill_confirm_rate,
            "min_random_win_margin": args.min_random_win_margin,
            "min_random_composite_margin": args.min_random_composite_margin,
            "min_scripted_win_ratio": args.min_scripted_win_ratio,
            "min_ppo_win_ratio": args.min_ppo_win_ratio,
            "min_ppo_composite_ratio": args.min_ppo_composite_ratio,
        },
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote_qrdqn_ready_json={out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 10


def _threshold_failures(args: argparse.Namespace, summary: dict[str, Any], tactical: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if _num(summary, "win_rate") < args.min_win_rate:
        failures.append(f"win_rate<{args.min_win_rate}")
    if _num(summary, "composite_score") < args.min_composite_score:
        failures.append(f"composite_score<{args.min_composite_score}")
    if _num(summary, "survival_rate") < args.min_survival_rate:
        failures.append(f"survival_rate<{args.min_survival_rate}")
    if _num(tactical, "bad_pass_rate") > args.max_bad_pass_rate:
        failures.append(f"bad_pass_rate>{args.max_bad_pass_rate}")
    if _num(tactical, "wasted_heal_rate") > args.max_wasted_heal_rate:
        failures.append(f"wasted_heal_rate>{args.max_wasted_heal_rate}")
    if _num(tactical, "critical_heal_miss_rate") > args.max_critical_heal_miss_rate:
        failures.append(f"critical_heal_miss_rate>{args.max_critical_heal_miss_rate}")
    if _num(tactical, "kill_confirm_opportunities") > 0 and _num(tactical, "kill_confirm_rate") < args.min_kill_confirm_rate:
        failures.append(f"kill_confirm_rate<{args.min_kill_confirm_rate}")
    return failures


def _num(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(value: float, reference: float) -> float:
    if reference <= 0:
        return 1.0 if value >= reference else 0.0
    return value / reference


if __name__ == "__main__":
    raise SystemExit(main())

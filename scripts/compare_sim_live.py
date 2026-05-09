# Compares live legal actions with simulator legal actions.
# Highlights where training environment rules diverge from the real DD2 mod snapshots.
# Consumes JSONL logs created by scripts/live_ppo.py.

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.dd_env import DarkestDungeonEnv
from env.live.mod_state import parse_state
from scripts.live_ppo import _build_mask_and_legal, _live_state_from_obs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare live legal actions against simulator legal actions for recorded frames.")
    p.add_argument("--log", default="runs/live_action_log.jsonl")
    p.add_argument("--out-json", default="runs/sim_live_diff.json")
    p.add_argument("--max-rows", type=int, default=0)
    return p.parse_args()


def _load_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("state"), dict):
                rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _obs_from_row(row: dict[str, Any]) -> dict[str, Any]:
    state = dict(row.get("state") or {})
    state["legal_actions"] = row.get("raw_legal_actions") or []
    if not state["legal_actions"]:
        state["legal_actions"] = [_raw_from_compact(a) for a in row.get("legal_actions") or []]
    return state


def _raw_from_compact(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("kind")
    if kind == "move":
        return {"move_delta": int(action.get("move_delta", 0) or 0), "move_skill_id": "move"}
    if kind == "pass":
        return {"pass_turn": True}
    if kind == "item":
        return {
            "item_id": action.get("item_id"),
            "target_idx": action.get("target_idx"),
            "target_team": action.get("target_side"),
        }
    return {
        "skill_idx": action.get("skill_idx"),
        "skill_id": action.get("skill_id"),
        "target_idx": action.get("target_idx"),
        "target_team": action.get("target_side"),
    }


def _spec_key(spec) -> tuple[Any, ...]:
    return (spec.kind, spec.skill_idx, spec.item_id, spec.target_idx, spec.target_side, spec.move_delta)


def main() -> int:
    args = parse_args()
    rows = _load_rows(Path(args.log), args.max_rows)
    env = DarkestDungeonEnv(seed=19)
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        obs = _obs_from_row(row)
        if obs.get("active_side") != "heroes":
            continue
        live_state = parse_state(_live_state_from_obs(obs))
        env.state = live_state
        _, live_legal = _build_mask_and_legal(env, obs)
        sim_legal = env.backend.legal_action_specs(live_state)
        live_keys = {_spec_key(a) for a in live_legal}
        sim_keys = {_spec_key(a) for a in sim_legal}
        only_live = sorted(live_keys - sim_keys)
        only_sim = sorted(sim_keys - live_keys)
        counts["frames"] += 1
        if only_live:
            counts["frames_with_only_live"] += 1
        if only_sim:
            counts["frames_with_only_sim"] += 1
        counts["only_live_actions"] += len(only_live)
        counts["only_sim_actions"] += len(only_sim)
        if (only_live or only_sim) and len(examples) < 50:
            examples.append(
                {
                    "row": idx,
                    "round": obs.get("round"),
                    "active_index": obs.get("active_index"),
                    "only_live": [list(x) for x in only_live[:20]],
                    "only_sim": [list(x) for x in only_sim[:20]],
                }
            )

    summary = {
        **{k: int(v) for k, v in sorted(counts.items())},
        "only_live_rate": counts["frames_with_only_live"] / max(1, counts["frames"]),
        "only_sim_rate": counts["frames_with_only_sim"] / max(1, counts["frames"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = {"summary": summary, "examples": examples}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote_sim_live_diff={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

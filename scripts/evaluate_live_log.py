# Replays recorded live decision frames through the current PPO model.
# Measures exact-match, bad pass, and critical-heal behavior against live logs.
# Uses live_ppo state conversion helpers to mirror live inference.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent
from env.dd_env import DarkestDungeonEnv
from env.live.mod_state import parse_state
from scripts.live_ppo import _adapt_obs_for_model, _build_mask_and_legal, _live_state_from_obs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate current PPO decisions on recorded live action frames.")
    p.add_argument("--log", default="runs/live_action_log.jsonl", help="JSONL produced by scripts/live_ppo.py --action-log.")
    p.add_argument("--model", default="runs/best/best_model.zip")
    p.add_argument("--out-json", default="", help="Optional path for detailed replay metrics.")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--critical-hp-threshold", type=float, default=0.25)
    p.add_argument("--fail-critical-heal-miss-rate", type=float, default=1.1)
    p.add_argument("--fail-bad-pass-rate", type=float, default=1.1)
    return p.parse_args()


def _load_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
    raw = row.get("raw_legal_actions") or [_raw_legal_action(a) for a in (row.get("legal_actions") or [])]
    state["legal_actions"] = raw
    state["items_available"] = state.get("items_available") or {}
    return state


def _raw_legal_action(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("kind")
    if kind == "pass":
        return {"pass_turn": True}
    if kind == "move":
        return {"move_delta": int(action.get("move_delta", 0) or 0)}
    if kind == "item":
        return {
            "item_id": str(action.get("item_id") or ""),
            "target_idx": int(action.get("target_idx", 0) or 0),
            "target_team": str(action.get("target_side") or "heroes"),
        }
    return {
        "skill_idx": int(action.get("skill_idx", 0) or 0),
        "skill_id": action.get("skill_id"),
        "target_idx": int(action.get("target_idx", 0) or 0),
        "target_team": str(action.get("target_side") or "enemies"),
    }


def _is_low_hp_heal_action(action: dict[str, Any], heroes: list[dict[str, Any]], threshold: float = 0.25) -> bool:
    kind = action.get("kind")
    skill_id = str(action.get("skill_id") or action.get("skillId") or "").lower()
    if kind == "skill":
        target_side = action.get("target_side")
    else:
        target_side = action.get("target_team")
    if "battlefield_medicine" in skill_id:
        pass
    elif int(action.get("skill_idx", -1) or -1) != 2:
        return False
    if target_side != "heroes":
        return False
    target_idx = int(action.get("target_idx", -1) or -1)
    if not (0 <= target_idx < len(heroes)):
        return False
    hero = heroes[target_idx]
    hp = int(hero.get("hp", 0) or 0)
    return hp <= 1 or (hp / max(1, int(hero.get("max_hp", 1) or 1))) <= threshold


def _compact_from_raw(env: DarkestDungeonEnv, raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("pass_turn"):
        idx = env._encode_action(env._decode_action(env.action_space.n - 1))
        return {"idx": idx, "kind": "pass"}
    if "move_delta" in raw:
        from env.data_model import ActionSpec

        spec = ActionSpec(kind="move", move_delta=int(raw.get("move_delta", 0) or 0))
        return {"idx": env._encode_action(spec), "kind": "move", "move_delta": spec.move_delta}
    if "item_id" in raw:
        return {
            "idx": -1,
            "kind": "item",
            "item_id": raw.get("item_id"),
            "skill_id": raw.get("skill_id"),
            "target_idx": int(raw.get("target_idx", 0) or 0),
            "target_side": str(raw.get("target_team") or "heroes"),
        }
    from env.data_model import ActionSpec

    spec = ActionSpec(
        kind="skill",
        skill_idx=int(raw.get("skill_idx", 0) or 0),
        target_idx=int(raw.get("target_idx", 0) or 0),
        target_side=str(raw.get("target_team") or "enemies"),
    )
    return {
        "idx": env._encode_action(spec),
        "kind": "skill",
        "skill_idx": spec.skill_idx,
        "skill_id": raw.get("skill_id"),
        "target_idx": spec.target_idx,
        "target_side": spec.target_side,
    }


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    rows = _load_rows(log_path, args.max_rows)
    env = DarkestDungeonEnv(seed=17)
    agent = PPOAgent.load(model_path, env=env)
    model_obs_dim = int(agent.model.observation_space.shape[0])

    total = 0
    exact_matches = 0
    pass_when_nonpass_legal = 0
    missed_low_hp_bfm = 0
    low_hp_bfm_frames = 0
    details: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        obs = _obs_from_row(row)
        if obs.get("active_side") != "heroes":
            continue
        live_state = parse_state(_live_state_from_obs(obs))
        env.state = live_state
        _, legal = _build_mask_and_legal(env, obs)
        if not legal:
            continue
        mask, _ = _build_mask_and_legal(env, obs)
        if not mask.any():
            continue
        obs_vec = _adapt_obs_for_model(env._obs(live_state), model_obs_dim)
        chosen_idx, _ = agent.predict(obs_vec, mask, deterministic=True)
        logged_idx = int(row.get("action_idx", -1) or -1)
        total += 1
        exact_matches += 1 if int(chosen_idx) == logged_idx else 0

        legal_compact = row.get("legal_actions") or []
        if row.get("raw_legal_actions"):
            legal_compact = [_compact_from_raw(env, a) for a in row.get("raw_legal_actions") or []]
        has_nonpass = any(a.get("kind") != "pass" for a in legal_compact)
        chosen_kind = env._decode_action(int(chosen_idx)).kind
        if chosen_kind == "pass" and has_nonpass:
            pass_when_nonpass_legal += 1

        heroes = list((row.get("state") or {}).get("heroes") or [])
        has_low_hp_bfm = any(_is_low_hp_heal_action(a, heroes, args.critical_hp_threshold) for a in legal_compact)
        if has_low_hp_bfm:
            low_hp_bfm_frames += 1
            chosen_action = next((a for a in legal_compact if int(a.get("idx", -999) or -999) == int(chosen_idx)), {})
            if not _is_low_hp_heal_action(chosen_action, heroes, args.critical_hp_threshold):
                missed_low_hp_bfm += 1

        details.append(
            {
                "row": idx,
                "logged_action_idx": logged_idx,
                "model_action_idx": int(chosen_idx),
                "exact_match": int(chosen_idx) == logged_idx,
                "has_low_hp_battlefield_medicine": has_low_hp_bfm,
            }
        )

    summary = {
        "rows_loaded": len(rows),
        "frames_evaluated": total,
        "exact_match_rate": exact_matches / max(1, total),
        "pass_when_nonpass_legal": pass_when_nonpass_legal,
        "low_hp_battlefield_medicine_frames": low_hp_bfm_frames,
        "missed_low_hp_battlefield_medicine": missed_low_hp_bfm,
        "critical_heal_miss_rate": missed_low_hp_bfm / max(1, low_hp_bfm_frames),
        "bad_pass_rate": pass_when_nonpass_legal / max(1, total),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote_metrics_json={out_path}")
    if summary["critical_heal_miss_rate"] > args.fail_critical_heal_miss_rate:
        return 2
    if summary["bad_pass_rate"] > args.fail_bad_pass_rate:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

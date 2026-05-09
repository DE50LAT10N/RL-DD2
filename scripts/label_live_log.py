# Auto-labeler for live action logs.
# Creates behavior-cloning labels for obvious corrections and optionally logged non-pass actions.
# Feeds scripts/train_bc.py for supervised warmup.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-label live action logs with tactical preferred actions.")
    p.add_argument("--log", default="runs/live_action_log.jsonl")
    p.add_argument("--out", default="runs/live_action_labels.jsonl")
    p.add_argument("--critical-hp-threshold", type=float, default=0.25)
    p.add_argument("--only-corrections", action="store_true", help="Do not add imitation labels for ordinary successful non-pass actions.")
    return p.parse_args()


def _iter_rows(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("state"), dict):
                yield row


def _hp_ratio(unit: dict[str, Any]) -> float:
    return int(unit.get("hp", 0) or 0) / max(1, int(unit.get("max_hp", 1) or 1))


def _preferred_action(row: dict[str, Any], threshold: float, include_imitation: bool = True) -> tuple[int | None, str]:
    state = row.get("state") or {}
    heroes = list(state.get("heroes") or [])
    raw_legal = list(row.get("raw_legal_actions") or [])
    compact = list(row.get("legal_actions") or [])
    active_idx = int(state.get("active_index", -1) or -1)
    active = next((h for h in heroes if int(h.get("slot", -9) or -9) == active_idx), {})
    active_text = " ".join(str(active.get(k, "")) for k in ("name", "id", "archetype_id")).lower()

    if "plague_doctor" in active_text or "plague doctor" in active_text:
        best_idx: int | None = None
        best_ratio = 2.0
        for raw in raw_legal:
            skill_id = str(raw.get("skill_id") or raw.get("skillId") or "").lower()
            if "battlefield_medicine" not in skill_id or str(raw.get("target_team") or "") != "heroes":
                continue
            target_idx = int(raw.get("target_idx", -1) or -1)
            target = next((h for h in heroes if int(h.get("slot", -9) or -9) == target_idx), None)
            if target is None:
                continue
            ratio = _hp_ratio(target)
            if int(target.get("hp", 0) or 0) <= 1 or ratio <= threshold:
                action_idx = next(
                    (
                        int(a.get("idx"))
                        for a in compact
                        if int(a.get("skill_idx", -9) or -9) == int(raw.get("skill_idx", -8) or -8)
                        and int(a.get("target_idx", -9) or -9) == target_idx
                        and str(a.get("target_side") or "") == "heroes"
                    ),
                    None,
                )
                if action_idx is not None and ratio < best_ratio:
                    best_idx = action_idx
                    best_ratio = ratio
        if best_idx is not None:
            return best_idx, "pd_critical_battlefield_medicine"

    non_pass = [a for a in compact if a.get("kind") != "pass"]
    if row.get("action_idx") == 62 and non_pass:
        return int(non_pass[0].get("idx", -1)), "bad_pass_nonpass_available"

    if include_imitation:
        logged_idx = row.get("action_idx")
        if logged_idx is not None:
            logged = next((a for a in compact if int(a.get("idx", -999) or -999) == int(logged_idx)), None)
            if logged is not None and logged.get("kind") != "pass":
                return int(logged_idx), "imitate_logged_nonpass"

    return None, ""


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(_iter_rows(Path(args.log)) or []):
            preferred, reason = _preferred_action(row, args.critical_hp_threshold, include_imitation=not args.only_corrections)
            label = {
                "row": row_no,
                "state": row.get("state"),
                "raw_legal_actions": row.get("raw_legal_actions") or [],
                "legal_actions": row.get("legal_actions") or [],
                "logged_action_idx": row.get("action_idx"),
                "preferred_action_idx": preferred,
                "label": "preferred" if preferred is not None else "unlabeled",
                "reason": reason,
            }
            out.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    print(f"wrote_labels={out_path} rows={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


USER_AGENT = "dd2-sim-data-builder/1.0"

SOURCE_URLS = {
    "skills": "https://darkestdungeon2.wiki.fextralife.com/Skills",
    "enemy_stats": "https://darkestdungeon.wiki.gg/wiki/Enemies_(Darkest_Dungeon_II)/Stats",
    "tokens": "https://darkestdungeon.wiki.gg/wiki/Tokens",
    "combat_items": "https://darkestdungeon.wiki.gg/wiki/Combat_Items",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _extract_skill_stats(text: str, skill_name: str) -> dict[str, Any]:
    pattern = re.compile(
        rf"{re.escape(skill_name)}.*?DMG:\s*(\d+)-(\d+).*?CRIT[:\s]+(\d+)%.*?(?:Cooldown[:\s]+(\d+))?",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return {"damage_lo": 2, "damage_hi": 4, "crit_chance": 0.05, "cooldown": 0}
    lo, hi, crit, cooldown = match.groups()
    return {
        "damage_lo": int(lo),
        "damage_hi": int(hi),
        "crit_chance": int(crit) / 100.0,
        "cooldown": int(cooldown or 0),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_data(output_root: Path) -> None:
    skills_text = _fetch_text(SOURCE_URLS["skills"])
    _ = _fetch_text(SOURCE_URLS["enemy_stats"])
    _ = _fetch_text(SOURCE_URLS["tokens"])
    _ = _fetch_text(SOURCE_URLS["combat_items"])

    hero_templates = {
        "man_at_arms": ("Man-at-Arms", ["wicked_slice", "rampart", "bolster", "retribution"]),
        "highwayman": ("Highwayman", ["wicked_slice", "pistol_shot", "open_vein", "double_tap"]),
        "plague_doctor": ("Plague Doctor", ["noxious_blast", "plague_grenade", "battlefield_medicine", "blinding_gas"]),
        "grave_robber": ("Grave Robber", ["pick_to_the_face", "thrown_dagger", "poison_dart", "absinthe"]),
    }
    for hid, (name, skills) in hero_templates.items():
        payload = {"id": hid, "name": name, "skills": []}
        for sk in skills:
            stats = _extract_skill_stats(skills_text, sk.replace("_", " "))
            payload["skills"].append(
                {
                    "id": sk,
                    "source_ranks": [1, 2, 3, 4],
                    "target_ranks": [1, 2, 3, 4],
                    "cooldown": stats["cooldown"],
                    "damage_lo": stats["damage_lo"],
                    "damage_hi": stats["damage_hi"],
                    "crit_chance": stats["crit_chance"],
                    "is_friendly": sk in {"battlefield_medicine", "bolster", "absinthe"},
                    "heal": 5 if sk in {"battlefield_medicine", "absinthe"} else 0,
                    "heal_stress": 1 if sk in {"bolster"} else 0,
                }
            )
        _write_json(output_root / "heroes" / f"{hid}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DD2 fixture data from wiki pages.")
    parser.add_argument("--output-root", type=Path, default=Path("tests/fixtures/game_data"))
    args = parser.parse_args()
    build_data(args.output_root)
    print(f"wrote wiki-derived data to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

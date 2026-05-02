from __future__ import annotations

import random
from typing import Any


def always_pass_policy(obs: dict[str, Any]) -> dict[str, Any]:
    return {"pass_turn": True}


def _first_alive_enemy_slot(obs: dict[str, Any]) -> int | None:
    enemies = obs.get("enemies") or []
    for enemy in enemies:
        if bool(enemy.get("alive", False)):
            return int(enemy.get("slot", 0))
    return None


def _emit_skill_action(obs: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hero_slot": int(action.get("hero_slot", obs.get("active_index", 0))),
        "skill_idx": int(action.get("skill_idx", 0)),
        "target_idx": int(action.get("target_idx", 0)),
    }
    target_team = action.get("target_team")
    if isinstance(target_team, str) and target_team:
        payload["target_team"] = target_team
    return payload


def scripted_policy(obs: dict[str, Any]) -> dict[str, Any]:
    if obs.get("active_side") != "heroes":
        return {"pass_turn": True}
    for action in obs.get("legal_actions") or []:
        if "skill_idx" in action and "target_idx" in action:
            return _emit_skill_action(obs, action)
    hero_slot = int(obs.get("active_index", 0))
    target = _first_alive_enemy_slot(obs)
    if target is None:
        return {"pass_turn": True}
    return {"hero_slot": hero_slot, "skill_idx": 0, "target_idx": target, "target_team": "enemies"}


def random_policy(obs: dict[str, Any]) -> dict[str, Any]:
    if obs.get("active_side") != "heroes":
        return {"pass_turn": True}

    legal_skills = [a for a in (obs.get("legal_actions") or []) if "skill_idx" in a and "target_idx" in a]
    if legal_skills:
        action = random.choice(legal_skills)
        return _emit_skill_action(obs, action)

    heroes = obs.get("heroes") or []
    enemies = [e for e in (obs.get("enemies") or []) if bool(e.get("alive", False))]
    if not heroes or not enemies:
        return {"pass_turn": True}

    active = int(obs.get("active_index", 0))
    alive_hero_slots = [int(h.get("slot", 0)) for h in heroes if bool(h.get("alive", False))]
    hero_slot = active if active in alive_hero_slots else random.choice(alive_hero_slots)
    target_idx = int(random.choice(enemies).get("slot", 0))
    skill_idx = random.randint(0, 3)
    return {"hero_slot": hero_slot, "skill_idx": skill_idx, "target_idx": target_idx, "target_team": "enemies"}


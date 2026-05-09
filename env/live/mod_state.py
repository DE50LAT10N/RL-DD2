# Parser for live DD2 mod state messages.
# Converts plugin JSON snapshots into the shared BattleState schema used by PPO inference.
# Handles tokens, deaths-door/death-armor, unit stats, and remnant detection.

from __future__ import annotations

from env.data_model import BattleState, Token, TokenId, Unit


def parse_state(msg: dict) -> BattleState:
    payload = msg.get("state", msg)
    heroes = [_parse_unit(u, "heroes") for u in payload.get("heroes", [])]
    enemies = [_parse_unit(u, "enemies") for u in payload.get("enemies", [])]
    rel: dict[tuple[str, str], int] = {}
    for r in payload.get("relationships", []):
        a = str(r.get("a", ""))
        b = str(r.get("b", ""))
        if a and b:
            rel[(a, b)] = int(r.get("value", 0))
    return BattleState(
        heroes=heroes,
        enemies=enemies,
        round=int(payload.get("round", 1)),
        initiative=[u.id for u in heroes + enemies],
        turn_idx=int(payload.get("active_index", 0)),
        skill_charges={},
        items_available={str(k): int(v) for k, v in payload.get("items_available", {}).items()},
        relationships=rel,
        done=bool(payload.get("done", False)),
        heroes_won=payload.get("heroes_won"),
    )


def _parse_unit(raw: dict, side: str) -> Unit:
    tokens: list[Token] = []
    for t in raw.get("tokens", []):
        tid = str(t.get("id", "BLOCK")).upper()
        if tid in TokenId.__members__:
            tokens.append(Token(TokenId[tid], int(t.get("count", 1))))
    if bool(raw.get("death_door", False)):
        tokens.append(Token(TokenId.DEATHS_DOOR, 1))
    death_armor = int(raw.get("death_armor", 0) or 0)
    if death_armor > 0:
        tokens.append(Token(TokenId.DEATH_ARMOR, death_armor))
    hp = int(raw.get("hp", 1))
    is_remnant = side == "enemies" and _is_enemy_remnant(raw)
    return Unit(
        id=str(raw.get("id") or raw.get("name") or "unit"),
        archetype_id=str(raw.get("archetype_id") or raw.get("name") or "unit"),
        side="heroes" if side == "heroes" else "enemies",
        rank=int(raw.get("rank", 1)),
        hp=hp,
        max_hp=int(raw.get("max_hp", max(1, hp))),
        speed=int(raw.get("speed", 5) or 5),
        size=max(1, int(raw.get("size", 1) or 1)),
        stress=int(raw.get("stress", 0)),
        tokens=tokens,
        alive=bool(raw.get("alive", hp > 0)),
        afflicted=bool(raw.get("afflicted", False)),
        is_remnant=is_remnant,
    )


def _is_enemy_remnant(unit: dict) -> bool:
    text = " ".join(
        str(unit.get(key, ""))
        for key in ("id", "name", "archetype_id", "display_name")
    ).lower()
    return any(
        marker in text
        for marker in (
            "corpse",
            "cadaver",
            "remnant",
            "tomb",
            "grave",
            "gravestone",
            "headstone",
            "РЎвЂљРЎР‚РЎС“Р С—",
            "Р Р…Р В°Р Т‘Р С–РЎР‚Р С•Р В±",
        )
    )

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
    hp = int(raw.get("hp", 1))
    is_remnant = side == "enemies" and _is_enemy_remnant(raw)
    return Unit(
        id=str(raw.get("id") or raw.get("name") or "unit"),
        archetype_id=str(raw.get("archetype_id") or raw.get("name") or "unit"),
        side="heroes" if side == "heroes" else "enemies",
        rank=int(raw.get("rank", 1)),
        hp=hp,
        max_hp=int(raw.get("max_hp", max(1, hp))),
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
            "С‚СЂСѓРї",
            "РЅР°РґРіСЂРѕР±",
        )
    )

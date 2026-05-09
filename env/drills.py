# Tactical drill state injectors for curriculum learning.
# Creates focused training states for healing, stress, movement, execution, and defensive decisions.
# Used by scripts/train.py without changing the base simulator API.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from env.data_model import BattleState, Token, TokenId


class DrillEnv(Protocol):
    state: BattleState
    backend: object

    def _obs(self, state: BattleState): ...


@dataclass(frozen=True)
class DrillResult:
    applied: bool
    drill_id: str = ""


DOT_TOKENS = (TokenId.BLEED, TokenId.BLIGHT, TokenId.BURN)


def _skills_for(env: DrillEnv, unit):
    return env.backend._skills_for(unit)  # type: ignore[attr-defined]


def _rng(env: DrillEnv):
    return env.backend.rng  # type: ignore[attr-defined]


def _set_active(env: DrillEnv, unit) -> None:
    env.state.speed_order = [unit.id] + [uid for uid in env.state.speed_order if uid != unit.id]
    env.state.initiative = list(env.state.speed_order)
    env.state.item_used_actor_id = None


def _find_hero(env: DrillEnv, archetype_id: str):
    return next((h for h in env.state.heroes if h.archetype_id == archetype_id and h.alive), None)


def _ensure_charges(env: DrillEnv, unit, skill_id: str) -> None:
    skill = next((sk for sk in _skills_for(env, unit) if sk.id == skill_id), None)
    if skill is not None and skill.charges > 0:
        env.state.skill_charges[(unit.id, skill.id)] = max(
            skill.charges,
            env.state.skill_charges.get((unit.id, skill.id), 0),
        )


def inject_pd_critical_heal(env: DrillEnv, *, max_hp_ratio: float = 0.25) -> DrillResult:
    pd = _find_hero(env, "plague_doctor")
    if pd is None or not any(sk.id == "battlefield_medicine" for sk in _skills_for(env, pd)):
        return DrillResult(False)
    victims = [h for h in env.state.heroes if h.id != pd.id and h.alive]
    if not victims:
        return DrillResult(False)
    victim = _rng(env).choice(victims)
    max_ratio_hp = max(1, int(victim.max_hp * max(0.05, min(0.5, float(max_hp_ratio)))))
    victim.hp = _rng(env).randint(1, max_ratio_hp)
    if _rng(env).random() < 0.35:
        dot = _rng(env).choice(DOT_TOKENS)
        victim.tokens.append(Token(id=dot, count=2, duration=3, amount=2))
    _ensure_charges(env, pd, "battlefield_medicine")
    _set_active(env, pd)
    return DrillResult(True, "pd_critical_heal")


def inject_pd_priority_heal(env: DrillEnv, *, max_hp_ratio: float = 0.25) -> DrillResult:
    pd = _find_hero(env, "plague_doctor")
    if pd is None or not any(sk.id == "battlefield_medicine" for sk in _skills_for(env, pd)):
        return DrillResult(False)
    victims = [h for h in env.state.heroes if h.id != pd.id and h.alive]
    if len(victims) < 2:
        return DrillResult(False)
    urgent, decoy = _rng(env).sample(victims, 2)
    urgent.hp = 1
    if not any(t.id == TokenId.DEATHS_DOOR for t in urgent.tokens):
        urgent.tokens.append(Token(id=TokenId.DEATHS_DOOR, count=1))
    decoy.hp = max(2, int(decoy.max_hp * max(0.28, min(0.48, float(max_hp_ratio) + 0.18))))
    for other in env.state.heroes:
        if other.id not in {pd.id, urgent.id, decoy.id} and other.alive:
            other.hp = max(other.hp, int(other.max_hp * 0.65))
    _ensure_charges(env, pd, "battlefield_medicine")
    _set_active(env, pd)
    return DrillResult(True, "pd_priority_heal")


def inject_pd_dot_cure(env: DrillEnv) -> DrillResult:
    pd = _find_hero(env, "plague_doctor")
    if pd is None or not any(sk.id == "battlefield_medicine" for sk in _skills_for(env, pd)):
        return DrillResult(False)
    victims = [h for h in env.state.heroes if h.alive]
    if not victims:
        return DrillResult(False)
    victim = _rng(env).choice(victims)
    dot = _rng(env).choice(DOT_TOKENS)
    victim.tokens.append(Token(id=dot, count=3, duration=3, amount=3))
    victim.hp = min(victim.hp, max(1, int(victim.max_hp * 0.55)))
    _ensure_charges(env, pd, "battlefield_medicine")
    _set_active(env, pd)
    return DrillResult(True, "pd_dot_cure")


def inject_maa_guard_deaths_door(env: DrillEnv) -> DrillResult:
    maa = _find_hero(env, "man_at_arms")
    if maa is None:
        return DrillResult(False)
    victims = [h for h in env.state.heroes if h.id != maa.id and h.alive]
    if not victims:
        return DrillResult(False)
    victim = _rng(env).choice(victims)
    victim.hp = 1
    if not any(t.id == TokenId.DEATHS_DOOR for t in victim.tokens):
        victim.tokens.append(Token(id=TokenId.DEATHS_DOOR, count=1))
    _set_active(env, maa)
    return DrillResult(True, "maa_guard_deaths_door")


def inject_maa_stress_bolster(env: DrillEnv) -> DrillResult:
    maa = _find_hero(env, "man_at_arms")
    if maa is None or not any(sk.id == "bolster" for sk in _skills_for(env, maa)):
        return DrillResult(False)
    victims = [h for h in env.state.heroes if h.id != maa.id and h.alive]
    if not victims:
        return DrillResult(False)
    urgent = _rng(env).choice(victims)
    urgent.stress = _rng(env).randint(8, 10)
    decoys = [h for h in victims if h.id != urgent.id]
    if decoys:
        decoy = _rng(env).choice(decoys)
        decoy.stress = _rng(env).randint(4, 6)
    for other in env.state.heroes:
        if other.id not in {maa.id, urgent.id} and other.alive:
            other.stress = min(other.stress, 6)
    _set_active(env, maa)
    return DrillResult(True, "maa_stress_bolster")


def inject_hellion_winded_tradeoff(env: DrillEnv) -> DrillResult:
    hellion = _find_hero(env, "hellion")
    if hellion is None:
        return DrillResult(False)
    hellion.tokens.append(Token(id=TokenId.WINDED, count=2, duration=3))
    hellion.hp = min(hellion.hp, max(1, int(hellion.max_hp * 0.45)))
    _set_active(env, hellion)
    return DrillResult(True, "hellion_winded_tradeoff")


def inject_hwm_execution_finish(env: DrillEnv) -> DrillResult:
    hwm = _find_hero(env, "highwayman")
    if hwm is None:
        return DrillResult(False)
    targets = [e for e in env.state.enemies if e.alive]
    if not targets:
        return DrillResult(False)
    target = _rng(env).choice(targets)
    target.hp = 1
    if not any(t.id == TokenId.DEATHS_DOOR for t in target.tokens):
        target.tokens.append(Token(id=TokenId.DEATHS_DOOR, count=1))
    _set_active(env, hwm)
    return DrillResult(True, "hwm_execution_finish")


def inject_move_to_valid_rank(env: DrillEnv) -> DrillResult:
    movable = [
        h for h in env.state.heroes
        if h.alive and not any(
            (not sk.is_friendly) and h.rank in sk.source_ranks
            for sk in _skills_for(env, h)
        )
    ]
    if not movable:
        movable = [h for h in env.state.heroes if h.alive]
    if not movable:
        return DrillResult(False)
    unit = _rng(env).choice(movable)
    unit.rank = 4 if unit.rank <= 2 else 1
    env.state.heroes.sort(key=lambda u: u.rank)
    _set_active(env, unit)
    return DrillResult(True, "move_to_valid_rank")


DRILLS = {
    "pd_critical_heal": inject_pd_critical_heal,
    "pd_priority_heal": inject_pd_priority_heal,
    "pd_dot_cure": inject_pd_dot_cure,
    "maa_guard_deaths_door": inject_maa_guard_deaths_door,
    "maa_stress_bolster": inject_maa_stress_bolster,
    "hellion_winded_tradeoff": inject_hellion_winded_tradeoff,
    "hwm_execution_finish": inject_hwm_execution_finish,
    "move_to_valid_rank": inject_move_to_valid_rank,
}

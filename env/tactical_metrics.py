# Evaluation metrics for tactical decision quality.
# Tracks bad passes, wasted heals, critical healing, remnant attacks, and kill-confirm behavior.
# Used by scripts/evaluate.py and pre-live readiness checks.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from env.data_model import ActionSpec, BattleState


@dataclass
class TacticalMetrics:
    counts: Counter[str] = field(default_factory=Counter)

    def observe_decision(
        self,
        state: BattleState,
        legal: list[ActionSpec],
        chosen: ActionSpec,
        trace: Any | None = None,
        after_state: BattleState | None = None,
    ) -> None:
        self.counts["decisions"] += 1
        if chosen.kind == "pass" and any(a.kind != "pass" for a in legal):
            self.counts["bad_pass"] += 1
        if chosen.kind == "move":
            self.counts["move_actions"] += 1

        backend = getattr(self, "_backend", None)
        critical_bfm = critical_battlefield_medicine_actions(state, legal, backend=backend)
        if critical_bfm:
            self.counts["critical_heal_opportunities"] += 1
            if any(_same_action(chosen, a) for a in critical_bfm):
                self.counts["critical_heal_success"] += 1
            else:
                self.counts["critical_heal_missed"] += 1

        if _is_wasted_heal(state, chosen, backend):
            self.counts["wasted_heal"] += 1
        if _is_attack_on_remnant(state, chosen):
            self.counts["corpse_attack"] += 1
        if _has_kill_confirm(state, legal):
            self.counts["kill_confirm_opportunities"] += 1
            if _low_enemy_killed(state, after_state) or (
                bool(getattr(trace, "damage_to_enemies", 0) or 0) and _trace_killed_enemy(state, trace)
            ):
                self.counts["kill_confirm_success"] += 1

    def summary(self) -> dict[str, float]:
        c = self.counts
        decisions = max(1, c["decisions"])
        critical = max(1, c["critical_heal_opportunities"])
        kill_confirm = max(1, c["kill_confirm_opportunities"])
        out = {k: float(v) for k, v in sorted(c.items())}
        out.update(
            {
                "bad_pass_rate": c["bad_pass"] / decisions,
                "wasted_heal_rate": c["wasted_heal"] / decisions,
                "critical_heal_success_rate": c["critical_heal_success"] / critical,
                "critical_heal_miss_rate": c["critical_heal_missed"] / critical,
                "kill_confirm_rate": c["kill_confirm_success"] / kill_confirm,
            }
        )
        return out


def composite_score(mean_reward: float, metrics: dict[str, float]) -> float:
    return (
        float(mean_reward)
        + 20.0 * float(metrics.get("critical_heal_success_rate", 0.0))
        + 8.0 * float(metrics.get("kill_confirm_rate", 0.0))
        - 15.0 * float(metrics.get("bad_pass_rate", 0.0))
        - 10.0 * float(metrics.get("wasted_heal_rate", 0.0))
        - 4.0 * float(metrics.get("critical_heal_miss_rate", 0.0))
    )


def critical_battlefield_medicine_actions(
    state: BattleState,
    legal: list[ActionSpec],
    threshold: float = 0.25,
    backend: Any | None = None,
) -> list[ActionSpec]:
    side, actor_idx, actor = _active_unit(state)
    if side != "heroes" or actor.archetype_id != "plague_doctor":
        return []
    out: list[ActionSpec] = []
    for action in legal:
        if action.kind != "skill" or action.target_side != "heroes":
            continue
        skill = _backend_skill_for(state, actor, action.skill_idx, backend)
        if skill is None or skill.id != "battlefield_medicine":
            continue
        target = _unit_at(state.heroes, action.target_idx)
        if target is None or not target.alive:
            continue
        if target.hp <= 1 or (target.hp / max(1, target.max_hp)) <= threshold:
            out.append(action)
    return out


def _active_unit(state: BattleState):
    for uid in list(state.speed_order):
        for pool_name, pool in (("heroes", state.heroes), ("enemies", state.enemies)):
            for idx, unit in enumerate(pool):
                if unit.id == uid and unit.alive and not unit.is_remnant:
                    return pool_name, idx, unit
    for idx, unit in enumerate(state.heroes):
        if unit.alive and not unit.is_remnant:
            return "heroes", idx, unit
    return "none", -1, None


def attach_skill_lookup(metrics: TacticalMetrics, backend: Any) -> TacticalMetrics:
    metrics._backend = backend  # type: ignore[attr-defined]
    return metrics


def _backend_skill_for(state: BattleState, actor: Any, skill_idx: int | None, backend: Any | None):
    if actor is None or skill_idx is None or backend is None:
        return None
    skills = backend._skills_for(actor)
    if not (0 <= int(skill_idx) < len(skills)):
        return None
    return skills[int(skill_idx)]


def _unit_at(units: list[Any], idx: int | None):
    if idx is None or not (0 <= int(idx) < len(units)):
        return None
    return units[int(idx)]


def _same_action(a: ActionSpec, b: ActionSpec) -> bool:
    return (
        a.kind == b.kind
        and a.skill_idx == b.skill_idx
        and a.item_id == b.item_id
        and a.target_idx == b.target_idx
        and a.target_side == b.target_side
        and a.move_delta == b.move_delta
    )


def _is_wasted_heal(state: BattleState, chosen: ActionSpec, backend: Any | None = None) -> bool:
    if chosen.kind != "skill" or chosen.target_side != "heroes":
        return False
    _, _, actor = _active_unit(state)
    skill = _backend_skill_for(state, actor, chosen.skill_idx, backend)
    if skill is not None and not (
        getattr(skill, "heal", 0) > 0
        or getattr(skill, "heal_percent", 0.0) > 0
        or bool(getattr(skill, "cures_tokens", []))
    ):
        return False
    target = _unit_at(state.heroes, chosen.target_idx)
    if target is None:
        return False
    return (target.hp / max(1, target.max_hp)) > 0.75


def _is_attack_on_remnant(state: BattleState, chosen: ActionSpec) -> bool:
    if chosen.kind != "skill" or chosen.target_side != "enemies":
        return False
    target = _unit_at(state.enemies, chosen.target_idx)
    return bool(target is not None and getattr(target, "is_remnant", False))


def _has_kill_confirm(state: BattleState, legal: list[ActionSpec]) -> bool:
    for action in legal:
        if action.kind == "skill" and action.target_side == "enemies":
            target = _unit_at(state.enemies, action.target_idx)
            if target is not None and target.alive and target.hp <= max(2, int(target.max_hp * 0.15)):
                return True
    return False


def _trace_killed_enemy(before: BattleState, trace: Any | None) -> bool:
    target_names = set(getattr(trace, "target_names", []) or [])
    return any(e.id in target_names and e.alive and e.hp <= max(2, int(e.max_hp * 0.15)) for e in before.enemies)


def _low_enemy_killed(before: BattleState, after: BattleState | None) -> bool:
    if after is None:
        return False
    after_by_id = {e.id: e for e in after.enemies}
    for enemy in before.enemies:
        if not enemy.alive or getattr(enemy, "is_remnant", False):
            continue
        if enemy.hp > max(2, int(enemy.max_hp * 0.15)):
            continue
        updated = after_by_id.get(enemy.id)
        if updated is None or not updated.alive or updated.hp <= 0:
            return True
    return False

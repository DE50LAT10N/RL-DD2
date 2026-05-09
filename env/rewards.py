# Reward shaping for tactical PPO training.
# Scores combat progress, survival, healing priority, stress control, movement discipline, and stalls.
# Loads tunable weights from configs/reward.yaml and consumes ActionTrace from env.engine.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from env.data_model import TokenId


@dataclass(slots=True)
class RewardWeights:
    win: float = 12.0
    defeat: float = -12.0
    damage_dealt: float = 0.35
    damage_progress_bonus: float = 0.08
    damage_taken: float = -0.3
    stress_saved: float = 0.1
    hero_alive_bonus: float = 0.3
    affliction_penalty: float = -0.4
    turn_penalty: float = -0.05
    enemy_kill_bonus: float = 4.0
    hero_death_penalty: float = -6.0
    stall_penalty: float = -0.25
    defensive_stall_penalty: float = -0.35
    heal_stall_penalty: float = -0.25
    pass_penalty: float = -0.9
    move_penalty: float = -0.08
    consecutive_move_penalty: float = -0.35
    move_loop_escalation: float = -0.2
    move_loop_penalty_cap: float = -1.2
    unnecessary_move_penalty: float = -1.0
    move_when_attack_available_penalty: float = -1.5
    diversity_bonus: float = 0.04
    no_effect_skill_penalty: float = -1.25
    repeat_same_action_penalty: float = -0.5
    phi_alpha: float = 0.5
    phi_beta: float = 0.3
    phi_gamma: float = 0.5
    phi_delta: float = 0.3
    phi_eps: float = 0.2
    heal_bonus: float = 0.16
    heal_critical_bonus: float = 0.08
    heal_bonus_cap: float = 1.25
    heal_low_hp_threshold: float = 0.5
    emergency_heal_hp_threshold: float = 0.25
    heal_low_hp_target_bonus: float = 1.0
    emergency_heal_bonus: float = 2.5
    plague_doctor_low_hp_heal_bonus: float = 1.2
    plague_doctor_ignore_critical_ally_penalty: float = -1.5
    plague_doctor_correct_critical_target_bonus: float = 3.0
    plague_doctor_wrong_heal_target_penalty: float = -3.0
    plague_doctor_missed_best_heal_target_penalty: float = -1.5
    high_stress_threshold: int = 7
    maa_bolster_high_stress_bonus: float = 2.5
    maa_bolster_wrong_target_penalty: float = -2.5
    maa_bolster_missed_best_stress_penalty: float = -1.25
    maa_ignore_high_stress_penalty: float = -1.0
    heal_healthy_target_penalty: float = -0.25
    wasted_heal_penalty: float = -0.5
    corpse_attack_penalty: float = -0.6
    missed_kill_penalty: float = -0.45
    unnecessary_defense_penalty: float = -0.35
    dot_cure_bonus: float = 0.35
    combo_consume_bonus: float = 0.18
    positive_relationship_bonus: float = 0.12
    negative_relationship_penalty: float = -0.18
    defense_setup_bonus: float = 0.18
    defense_critical_bonus: float = 0.22
    defense_bonus_cap: float = 1.0


def load_reward_weights(path: Path | None = None) -> RewardWeights:
    p = path or (Path(__file__).resolve().parent.parent / "configs" / "reward.yaml")
    if not p.exists():
        return RewardWeights()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return RewardWeights(**{k: data.get(k, getattr(RewardWeights(), k)) for k in RewardWeights.__dataclass_fields__.keys()})


class RewardCalculator:
    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.w = weights or load_reward_weights()
        self._last_skill_name: str | None = None
        self._last_action_key: tuple[str, str, tuple[str, ...]] | None = None
        self._consecutive_moves = 0

    def reset(self) -> None:
        self._last_skill_name = None
        self._last_action_key = None
        self._consecutive_moves = 0

    def _phi(self, state: Any) -> float:
        hero_hp_now = sum(max(0, u.hp) for u in state.heroes)
        hero_hp_cap = sum(max(1, u.max_hp) for u in state.heroes)
        enemy_hp_now = self._hp_sum(state.enemies)
        enemy_hp_cap = self._hp_cap(state.enemies)
        alive_heroes = sum(1 for u in state.heroes if u.alive)
        alive_enemies = self._alive_count(state.enemies)
        stress = sum(max(0, u.stress) for u in state.heroes)
        stress_cap = max(1, len(state.heroes) * 10)
        hp_heroes_pct = hero_hp_now / max(1, hero_hp_cap)
        hp_enemies_pct = enemy_hp_now / max(1, enemy_hp_cap)
        alive_heroes_pct = alive_heroes / max(1, len(state.heroes))
        alive_enemies_pct = alive_enemies / max(1, len(self._combat_units(state.enemies)))
        stress_pct = stress / stress_cap
        return (
            self.w.phi_alpha * hp_heroes_pct
            + self.w.phi_beta * alive_heroes_pct
            - self.w.phi_gamma * hp_enemies_pct
            - self.w.phi_delta * alive_enemies_pct
            - self.w.phi_eps * stress_pct
        )

    @staticmethod
    def _token_count(unit: Any, token: TokenId) -> int:
        return sum(t.count for t in getattr(unit, "tokens", []) if t.id == token)

    @staticmethod
    def _combat_units(units: list[Any]) -> list[Any]:
        # Training treats active remnants/tombstones as board blockers that must
        # be removed, so reward progress must include their HP and death.
        return list(units)

    @classmethod
    def _hp_sum(cls, units: list[Any]) -> int:
        return sum(max(0, int(getattr(u, "hp", 0))) for u in cls._combat_units(units))

    @classmethod
    def _hp_cap(cls, units: list[Any]) -> int:
        return sum(max(1, int(getattr(u, "max_hp", 1))) for u in cls._combat_units(units))

    @classmethod
    def _alive_count(cls, units: list[Any]) -> int:
        return sum(1 for u in cls._combat_units(units) if bool(getattr(u, "alive", False)))

    @staticmethod
    def _actor_is(actor_name: str, archetype_id: str) -> bool:
        return archetype_id in str(actor_name)

    def compute(self, before: Any, after: Any, trace: Any) -> float:
        done_reward = 0.0
        if after.done:
            done_reward = self.w.win if after.heroes_won else self.w.defeat
        hero_hp_before = sum(max(0, u.hp) for u in before.heroes)
        hero_hp_after = sum(max(0, u.hp) for u in after.heroes)
        enemy_hp_before = self._hp_sum(before.enemies)
        enemy_hp_after = self._hp_sum(after.enemies)
        stress_before = sum(max(0, u.stress) for u in before.heroes)
        stress_after = sum(max(0, u.stress) for u in after.heroes)
        enemy_alive_before = self._alive_count(before.enemies)
        enemy_alive_after = self._alive_count(after.enemies)
        hero_alive_before = sum(1 for u in before.heroes if u.alive)
        alive_after = sum(1 for u in after.heroes if u.alive)
        afflicted_after = sum(1 for u in after.heroes if u.afflicted)
        enemy_kills = max(0, enemy_alive_before - enemy_alive_after)
        hero_losses = max(0, hero_alive_before - alive_after)
        low_hp_threshold = float(self.w.heal_low_hp_threshold)
        emergency_hp_threshold = float(self.w.emergency_heal_hp_threshold)
        low_hp_before = sum(1 for u in before.heroes if u.alive and (u.hp / max(1, u.max_hp)) <= low_hp_threshold)
        low_hp_after = sum(1 for u in after.heroes if u.alive and (u.hp / max(1, u.max_hp)) <= low_hp_threshold)
        emergency_hp_before = [
            u for u in before.heroes
            if u.alive and (u.hp <= 1 or (u.hp / max(1, u.max_hp)) <= emergency_hp_threshold)
        ]
        most_urgent_ally = min(
            emergency_hp_before,
            key=lambda u: (u.hp / max(1, u.max_hp), u.hp),
            default=None,
        )
        high_stress_threshold = int(self.w.high_stress_threshold)
        high_stress_before = [
            u for u in before.heroes
            if u.alive and int(getattr(u, "stress", 0) or 0) >= high_stress_threshold
        ]
        most_stressed_ally = max(
            high_stress_before,
            key=lambda u: (int(getattr(u, "stress", 0) or 0), -u.hp / max(1, u.max_hp)),
            default=None,
        )
        target_names = set(getattr(trace, "target_names", []) or [])
        skill_name = getattr(trace, "skill_name", None)
        actor_name = str(getattr(trace, "actor_name", ""))

        # Penalize turns that do not advance combat state.
        did_progress = (enemy_hp_before - enemy_hp_after) > 0 or enemy_kills > 0
        enemy_hp_delta = enemy_hp_before - enemy_hp_after

        shaped = self._phi(after) - self._phi(before)
        reward = (
            enemy_hp_delta * self.w.damage_dealt
            + max(0, enemy_hp_delta) * self.w.damage_progress_bonus
            + (hero_hp_before - hero_hp_after) * self.w.damage_taken
            + (stress_before - stress_after) * self.w.stress_saved
            + enemy_kills * self.w.enemy_kill_bonus
            + hero_losses * self.w.hero_death_penalty
            + alive_after * self.w.hero_alive_bonus
            + afflicted_after * self.w.affliction_penalty
            + self.w.turn_penalty
            + (0.0 if did_progress else self.w.stall_penalty)
            + shaped
            + done_reward
        )
        healing_done = max(0, int(getattr(trace, "healing_done", 0) or 0))
        tokens_cured = max(0, int(getattr(trace, "tokens_cured", 0) or 0))
        if healing_done > 0:
            heal_bonus = min(self.w.heal_bonus_cap, healing_done * self.w.heal_bonus)
            # Reward healing more when it removes critical HP states.
            critical_relief = max(0, low_hp_before - low_hp_after)
            heal_bonus += critical_relief * self.w.heal_critical_bonus
            target_names = set(getattr(trace, "target_names", []) or [])
            healed_targets_before = [u for u in before.heroes if u.id in target_names and u.alive]
            healed_low_hp_targets = [
                u for u in healed_targets_before
                if (u.hp / max(1, u.max_hp)) <= low_hp_threshold
            ]
            healed_emergency_targets = [
                u for u in healed_targets_before
                if u.hp <= 1 or (u.hp / max(1, u.max_hp)) <= emergency_hp_threshold
            ]
            if healed_low_hp_targets:
                missing_ratio = max(1.0 - (u.hp / max(1, u.max_hp)) for u in healed_low_hp_targets)
                heal_bonus += self.w.heal_low_hp_target_bonus * missing_ratio
                if getattr(trace, "skill_name", "") == "battlefield_medicine":
                    heal_bonus += self.w.plague_doctor_low_hp_heal_bonus * missing_ratio
            if healed_emergency_targets:
                missing_ratio = max(1.0 - (u.hp / max(1, u.max_hp)) for u in healed_emergency_targets)
                heal_bonus += self.w.emergency_heal_bonus * missing_ratio
            elif healed_targets_before:
                heal_bonus += self.w.heal_healthy_target_penalty
            if not did_progress and not after.done:
                heal_bonus *= 0.6
            reward += min(self.w.heal_bonus_cap, heal_bonus)
            if not did_progress and not after.done and low_hp_before <= 0:
                reward += self.w.heal_stall_penalty
        if self._actor_is(actor_name, "plague_doctor") and emergency_hp_before and not after.done:
            critical_target_ids = {u.id for u in emergency_hp_before}
            hit_critical_target = bool(target_names & critical_target_ids)
            hit_best_target = bool(most_urgent_ally is not None and most_urgent_ally.id in target_names)
            if skill_name == "battlefield_medicine":
                if hit_critical_target:
                    healed = [u for u in emergency_hp_before if u.id in target_names]
                    missing_ratio = max((1.0 - (u.hp / max(1, u.max_hp)) for u in healed), default=1.0)
                    reward += self.w.plague_doctor_correct_critical_target_bonus * missing_ratio
                    if most_urgent_ally is not None and not hit_best_target:
                        reward += self.w.plague_doctor_missed_best_heal_target_penalty
                else:
                    reward += self.w.plague_doctor_wrong_heal_target_penalty
            elif enemy_kills <= 0:
                reward += self.w.plague_doctor_ignore_critical_ally_penalty
        if self._actor_is(actor_name, "man_at_arms") and high_stress_before and not after.done:
            high_stress_ids = {u.id for u in high_stress_before}
            hit_high_stress = bool(target_names & high_stress_ids)
            hit_best_stress = bool(most_stressed_ally is not None and most_stressed_ally.id in target_names)
            if skill_name == "bolster":
                if hit_high_stress:
                    stress_ratio = max(
                        (int(getattr(u, "stress", 0) or 0) / 10.0 for u in high_stress_before if u.id in target_names),
                        default=1.0,
                    )
                    reward += self.w.maa_bolster_high_stress_bonus * stress_ratio
                    if most_stressed_ally is not None and not hit_best_stress:
                        reward += self.w.maa_bolster_missed_best_stress_penalty
                else:
                    reward += self.w.maa_bolster_wrong_target_penalty
            elif enemy_kills <= 0:
                reward += self.w.maa_ignore_high_stress_penalty
        if tokens_cured > 0:
            reward += tokens_cured * self.w.dot_cure_bonus
        if healing_done > 0:
            healed_targets_before = [u for u in before.heroes if u.id in target_names and u.alive]
            if healed_targets_before and all((u.hp / max(1, u.max_hp)) > 0.75 for u in healed_targets_before):
                reward += self.w.wasted_heal_penalty
        if any(e.id in target_names and getattr(e, "is_remnant", False) for e in before.enemies):
            reward += self.w.corpse_attack_penalty
        low_enemy_before = [
            e for e in before.enemies
            if e.alive and not getattr(e, "is_remnant", False) and e.hp <= max(2, int(e.max_hp * 0.15))
        ]
        if low_enemy_before and enemy_kills <= 0 and not emergency_hp_before and skill_name not in {"battlefield_medicine", "item"}:
            reward += self.w.missed_kill_penalty
        if bool(getattr(trace, "combo_consumed", False)):
            reward += self.w.combo_consume_bonus
        relationship_actout = str(getattr(trace, "relationship_actout", "") or "")
        if relationship_actout == "positive":
            reward += self.w.positive_relationship_bonus
        elif relationship_actout == "negative":
            reward += self.w.negative_relationship_penalty

        defensive_tokens = (TokenId.BLOCK, TokenId.BLOCK_PLUS, TokenId.TAUNT)
        defense_gained = 0
        for i, before_unit in enumerate(before.heroes):
            if i >= len(after.heroes):
                break
            after_unit = after.heroes[i]
            for token in defensive_tokens:
                delta = self._token_count(after_unit, token) - self._token_count(before_unit, token)
                if delta > 0:
                    defense_gained += delta
        if defense_gained > 0:
            defense_bonus = min(self.w.defense_bonus_cap, defense_gained * self.w.defense_setup_bonus)
            if low_hp_before > 0:
                defense_bonus += self.w.defense_critical_bonus
            if not did_progress and not after.done:
                defense_bonus *= 0.5
                reward += self.w.defensive_stall_penalty
            reward += min(self.w.defense_bonus_cap, defense_bonus)

        if skill_name == "pass":
            reward += self.w.pass_penalty
        elif skill_name == "move":
            reward += self.w.move_penalty
            if int(getattr(trace, "legal_non_move_actions", 0) or 0) > 0:
                reward += self.w.unnecessary_move_penalty
            if int(getattr(trace, "legal_offensive_actions", 0) or 0) > 0:
                reward += self.w.move_when_attack_available_penalty
            if self._last_skill_name == "move":
                self._consecutive_moves += 1
                loop_penalty = self.w.consecutive_move_penalty + self.w.move_loop_escalation * max(0, self._consecutive_moves - 2)
                reward += max(self.w.move_loop_penalty_cap, loop_penalty)
            else:
                self._consecutive_moves = 1
        elif skill_name and skill_name not in {"item", "stunned"}:
            self._consecutive_moves = 0
            useful_effect = (
                int(getattr(trace, "damage_to_enemies", 0) or 0) > 0
                or int(getattr(trace, "damage_to_heroes", 0) or 0) > 0
                or int(getattr(trace, "healing_done", 0) or 0) > 0
                or int(getattr(trace, "tokens_cured", 0) or 0) > 0
                or int(getattr(trace, "stress_delta", 0) or 0) != 0
                or int(getattr(trace, "relationships_delta", 0) or 0) != 0
            )
            if not useful_effect:
                reward += self.w.no_effect_skill_penalty
        elif skill_name:
            self._consecutive_moves = 0
        if skill_name and self._last_skill_name and skill_name != self._last_skill_name:
            reward += self.w.diversity_bonus
        if skill_name:
            action_key = (
                str(getattr(trace, "actor_name", "")),
                str(skill_name),
                tuple(str(x) for x in (getattr(trace, "target_names", []) or [])),
            )
            if action_key == self._last_action_key and skill_name not in {"pass", "move", "item", "stunned"}:
                reward += self.w.repeat_same_action_penalty
            self._last_action_key = action_key
        if skill_name:
            self._last_skill_name = skill_name
        return reward

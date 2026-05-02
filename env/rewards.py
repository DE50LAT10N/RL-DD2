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
    damage_taken: float = -0.3
    stress_saved: float = 0.1
    hero_alive_bonus: float = 0.3
    affliction_penalty: float = -0.4
    turn_penalty: float = -0.05
    enemy_kill_bonus: float = 4.0
    hero_death_penalty: float = -6.0
    stall_penalty: float = -0.25
    pass_penalty: float = -0.12
    diversity_bonus: float = 0.04
    phi_alpha: float = 0.5
    phi_beta: float = 0.3
    phi_gamma: float = 0.5
    phi_delta: float = 0.3
    phi_eps: float = 0.2
    heal_bonus: float = 0.16
    heal_critical_bonus: float = 0.08
    heal_bonus_cap: float = 1.25
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

    def _phi(self, state: Any) -> float:
        hero_hp_now = sum(max(0, u.hp) for u in state.heroes)
        hero_hp_cap = sum(max(1, u.max_hp) for u in state.heroes)
        enemy_hp_now = sum(max(0, u.hp) for u in state.enemies)
        enemy_hp_cap = sum(max(1, u.max_hp) for u in state.enemies)
        alive_heroes = sum(1 for u in state.heroes if u.alive)
        alive_enemies = sum(1 for u in state.enemies if u.alive)
        stress = sum(max(0, u.stress) for u in state.heroes)
        stress_cap = max(1, len(state.heroes) * 10)
        hp_heroes_pct = hero_hp_now / max(1, hero_hp_cap)
        hp_enemies_pct = enemy_hp_now / max(1, enemy_hp_cap)
        alive_heroes_pct = alive_heroes / max(1, len(state.heroes))
        alive_enemies_pct = alive_enemies / max(1, len(state.enemies))
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

    def compute(self, before: Any, after: Any, trace: Any) -> float:
        done_reward = 0.0
        if after.done:
            done_reward = self.w.win if after.heroes_won else self.w.defeat
        hero_hp_before = sum(max(0, u.hp) for u in before.heroes)
        hero_hp_after = sum(max(0, u.hp) for u in after.heroes)
        enemy_hp_before = sum(max(0, u.hp) for u in before.enemies)
        enemy_hp_after = sum(max(0, u.hp) for u in after.enemies)
        stress_before = sum(max(0, u.stress) for u in before.heroes)
        stress_after = sum(max(0, u.stress) for u in after.heroes)
        enemy_alive_before = sum(1 for u in before.enemies if u.alive)
        enemy_alive_after = sum(1 for u in after.enemies if u.alive)
        hero_alive_before = sum(1 for u in before.heroes if u.alive)
        alive_after = sum(1 for u in after.heroes if u.alive)
        afflicted_after = sum(1 for u in after.heroes if u.afflicted)
        enemy_kills = max(0, enemy_alive_before - enemy_alive_after)
        hero_losses = max(0, hero_alive_before - alive_after)
        low_hp_before = sum(1 for u in before.heroes if u.alive and (u.hp / max(1, u.max_hp)) <= 0.45)
        low_hp_after = sum(1 for u in after.heroes if u.alive and (u.hp / max(1, u.max_hp)) <= 0.45)

        # Penalize turns that do not advance combat state.
        did_progress = (enemy_hp_before - enemy_hp_after) > 0 or enemy_kills > 0

        shaped = self._phi(after) - self._phi(before)
        reward = (
            (enemy_hp_before - enemy_hp_after) * self.w.damage_dealt
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
        if healing_done > 0:
            heal_bonus = min(self.w.heal_bonus_cap, healing_done * self.w.heal_bonus)
            # Reward healing more when it removes critical HP states.
            critical_relief = max(0, low_hp_before - low_hp_after)
            heal_bonus += critical_relief * self.w.heal_critical_bonus
            if not did_progress and not after.done:
                heal_bonus *= 0.6
            reward += min(self.w.heal_bonus_cap, heal_bonus)

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
            reward += min(self.w.defense_bonus_cap, defense_bonus)

        skill_name = getattr(trace, "skill_name", None)
        if skill_name == "pass":
            reward += self.w.pass_penalty
        if skill_name and self._last_skill_name and skill_name != self._last_skill_name:
            reward += self.w.diversity_bonus
        if skill_name:
            self._last_skill_name = skill_name
        return reward

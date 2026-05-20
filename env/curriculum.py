# Shared curriculum environments for simulator training and holdout evaluation.
# Used by PPO now and by future off-policy agents that need the same schedule.

from __future__ import annotations

from env.dd_env import DarkestDungeonEnv
from env.drills import DRILLS, inject_pd_critical_heal, inject_pd_priority_heal


HOLDOUT_ENCOUNTERS = [
    "holdout_double_ghoul",
    "holdout_double_cultist",
    "holdout_mix_wave",
    "holdout_elite_pair",
    "holdout_elite_swarm",
]
PHASE1 = ["gaunt_pair", "lost_battalion_road", "road_fight"]
PHASE2 = ["gaunt_pair", "cultist_trio", "military_squad", "lost_battalion_road", "swine_pair", "road_fight"]
PHASE2B = PHASE2 + ["ghoul_solo"]
PHASE3_PAIR = PHASE2B + ["holdout_elite_pair"]
PHASE3_FULL = PHASE3_PAIR + ["holdout_elite_swarm"]
EASY_ENCOUNTERS = {"gaunt_pair", "lost_battalion_road", "road_fight"}


class CurriculumEnv(DarkestDungeonEnv):
    def __init__(
        self,
        *args,
        total_steps: int = 1_000_000,
        drill_ratio: float = 0.28,
        critical_heal_drill_ratio: float = 0.18,
        critical_heal_drill_max_hp_ratio: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.total_steps = total_steps
        self.drill_ratio = max(0.0, min(1.0, float(drill_ratio)))
        self.critical_heal_drill_ratio = max(0.0, min(1.0, float(critical_heal_drill_ratio)))
        self.critical_heal_drill_max_hp_ratio = max(0.05, min(0.5, float(critical_heal_drill_max_hp_ratio)))
        self.global_steps = 0
        self._current_encounter: str | None = None
        self._encounter_stats: dict[str, dict[str, int]] = {}
        self._easy_ratio = 0.2

    def _phase_encounters(self) -> list[str]:
        """Fraction of per-env steps; each parallel env advances its own counter."""
        t = max(1, int(self.total_steps))
        r = min(1.0, self.global_steps / t)
        if r < 0.22:
            return PHASE1
        if r < 0.55:
            return PHASE2
        if r < 0.72:
            return PHASE2B
        if r < 0.88:
            return PHASE3_PAIR
        return PHASE3_FULL

    def _choose_encounter(self) -> str:
        pool = self._phase_encounters()
        if not pool:
            return "road_fight"
        easy_pool = [x for x in pool if x in EASY_ENCOUNTERS]
        hard_pool = [x for x in pool if x not in EASY_ENCOUNTERS]
        if hard_pool and easy_pool and self.backend.rng.random() < self._easy_ratio:
            return self.backend.rng.choice(easy_pool)

        weighted_pool = hard_pool or pool
        weights: list[float] = []
        for encounter_id in weighted_pool:
            row = self._encounter_stats.get(encounter_id)
            if not row or row["episodes"] < 3:
                weights.append(1.0)
                continue
            win_rate = row["wins"] / max(1, row["episodes"])
            weights.append(max(0.1, 2.4 - 2.6 * win_rate))
        return self.backend.rng.choices(weighted_pool, weights=weights, k=1)[0]

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if "encounter_id" not in opts:
            opts["encounter_id"] = self._choose_encounter()
        self._current_encounter = str(opts["encounter_id"])
        _, info = super().reset(seed=seed, options=opts)
        self._maybe_inject_drill()
        return self._obs(self.state), info

    def _maybe_inject_drill(self) -> None:
        if self.drill_ratio <= 0 or self.backend.rng.random() >= self.drill_ratio:
            return
        weighted = [
            ("pd_critical_heal", self.critical_heal_drill_ratio),
            ("pd_priority_heal", 0.18),
            ("pd_dot_cure", 0.16),
            ("maa_guard_deaths_door", 0.14),
            ("maa_stress_bolster", 0.14),
            ("hellion_winded_tradeoff", 0.12),
            ("hwm_execution_finish", 0.16),
            ("move_to_valid_rank", 0.14),
        ]
        total = sum(max(0.0, weight) for _, weight in weighted)
        if total <= 0:
            return
        roll = self.backend.rng.random() * total
        upto = 0.0
        for drill_id, weight in weighted:
            upto += max(0.0, weight)
            if roll <= upto:
                if drill_id == "pd_critical_heal":
                    inject_pd_critical_heal(self, max_hp_ratio=self.critical_heal_drill_max_hp_ratio)
                elif drill_id == "pd_priority_heal":
                    inject_pd_priority_heal(self, max_hp_ratio=self.critical_heal_drill_max_hp_ratio)
                else:
                    DRILLS[drill_id](self)
                return

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        self.global_steps += 1
        if terminated or truncated:
            encounter_id = self._current_encounter or "unknown"
            row = self._encounter_stats.setdefault(encounter_id, {"episodes": 0, "wins": 0})
            row["episodes"] += 1
            row["wins"] += 1 if info.get("heroes_won") else 0
            recent = [r["wins"] / max(1, r["episodes"]) for r in self._encounter_stats.values() if r["episodes"] >= 3]
            if recent:
                avg_win = sum(recent) / len(recent)
                if avg_win < 0.45:
                    self._easy_ratio = 0.35
                elif avg_win < 0.6:
                    self._easy_ratio = 0.28
                elif avg_win < 0.75:
                    self._easy_ratio = 0.18
                else:
                    self._easy_ratio = 0.1
        return obs, reward, terminated, truncated, info


class HoldoutEvalEnv(DarkestDungeonEnv):
    def __init__(
        self,
        *args,
        critical_heal_drill_ratio: float = 0.25,
        critical_heal_drill_max_hp_ratio: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.critical_heal_drill_ratio = max(0.0, min(1.0, float(critical_heal_drill_ratio)))
        self.critical_heal_drill_max_hp_ratio = max(0.05, min(0.5, float(critical_heal_drill_max_hp_ratio)))

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if "encounter_id" not in opts:
            opts["encounter_id"] = self.backend.rng.choice(HOLDOUT_ENCOUNTERS)
        _, info = super().reset(seed=seed, options=opts)
        if self.backend.rng.random() < self.critical_heal_drill_ratio:
            if self.backend.rng.random() < 0.5:
                inject_pd_priority_heal(self, max_hp_ratio=self.critical_heal_drill_max_hp_ratio)
            else:
                inject_pd_critical_heal(self, max_hp_ratio=self.critical_heal_drill_max_hp_ratio)
        return self._obs(self.state), info

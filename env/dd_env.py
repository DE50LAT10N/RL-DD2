# Gymnasium-compatible Darkest Dungeon II tactical environment.
# Encodes simulator state/actions for MaskablePPO training and evaluation.
# Depends on SimBattleBackend, RewardCalculator, and stable action-space constants used by live PPO.

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.data_model import ActionSpec, BattleState, TokenId
from env.engine import SimBattleBackend
from env.rewards import RewardCalculator

MAX_SKILLS_PER_UNIT = 5
ACTION_TARGET_SLOTS = 4
NUM_ITEMS = 2
ACTION_MOVE_DELTAS = (-3, -2, -1, 1, 2, 3)
MOVE_ACTIONS = len(ACTION_MOVE_DELTAS)
ACTION_SPACE_SIZE = MAX_SKILLS_PER_UNIT * (ACTION_TARGET_SLOTS * 2) + NUM_ITEMS * (ACTION_TARGET_SLOTS * 2) + MOVE_ACTIONS + 1
# One-hot style hint so policy can switch tactics (holdout elites vs road fights).
ENCOUNTER_HINT_DIM = 4


@lru_cache(maxsize=1)
def _default_training_party() -> tuple[str, ...] | None:
    """Roster from configs/training_party.json when present (MAA, Hellion, PD, HWM)."""
    path = Path(__file__).resolve().parent.parent / "configs" / "training_party.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ids = raw.get("hero_lineup") or raw.get("archetype_ids")
    if not isinstance(ids, list):
        return None
    out = tuple(str(x).strip() for x in ids if str(x).strip())
    return out[:4] or None


class DarkestDungeonEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 7, backend=None, max_episode_steps: int = 180) -> None:
        super().__init__()
        self.backend = backend or SimBattleBackend(seed=seed)
        self.reward_calculator = RewardCalculator()
        party = list(_default_training_party()) if _default_training_party() else None
        self.state: BattleState = self.backend.reset(seed=seed, hero_lineup=party)
        self._last_actions: list[ActionSpec] = []
        self.max_episode_steps = max_episode_steps
        self._episode_steps = 0
        token_count = len(TokenId)
        per_unit = 9 + token_count
        obs_dim = per_unit * 8 + 2 + NUM_ITEMS + MAX_SKILLS_PER_UNIT + ENCOUNTER_HINT_DIM + 16
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        opts = dict(options or {})
        encounter_id = opts.get("encounter_id")
        if "hero_lineup" in opts:
            hero_lineup = opts["hero_lineup"]
            hero_lineup = list(hero_lineup) if hero_lineup else None
        else:
            party = _default_training_party()
            hero_lineup = list(party) if party else None
        self.state = self.backend.reset(seed=seed, encounter_id=encounter_id, hero_lineup=hero_lineup)
        self.reward_calculator.reset()
        self._last_actions = self.backend.legal_action_specs(self.state)
        self._episode_steps = 0
        return self._obs(self.state), {}

    def step(self, action: int):
        before = copy.deepcopy(self.state)
        legal = self.backend.legal_action_specs(self.state)
        selected = next((a for a in legal if self._encode_action(a) == int(action)), self._decode_action(int(action)))
        if hasattr(self.backend, "execute_spec"):
            self.backend.execute_spec(self.state, selected)
        elif selected.kind == "skill":
            self.backend.execute_action(self.state, selected.skill_idx or 0, selected.target_idx or 0)
        else:
            self.state.turn_idx += 1

        self._episode_steps += 1
        self._last_actions = self.backend.legal_action_specs(self.state)
        reward = self.reward_calculator.compute(before, self.state, self.backend.last_trace)
        terminated = self.state.done
        truncated = (self._episode_steps >= self.max_episode_steps) and not terminated
        if truncated:
            enemy_hp_left = sum(max(0, int(getattr(u, "hp", 0))) for u in self.state.enemies if getattr(u, "alive", False))
            enemy_hp_cap = sum(max(1, int(getattr(u, "max_hp", 1))) for u in self.state.enemies)
            enemy_hp_ratio = enemy_hp_left / max(1, enemy_hp_cap)
            reward -= 6.0 + 6.0 * enemy_hp_ratio
        return self._obs(self.state), float(reward), terminated, truncated, {
            "heroes_won": self.state.heroes_won,
            "is_success": bool(self.state.heroes_won),
            "TimeLimit.truncated": truncated,
        }

    def action_masks(self) -> np.ndarray:
        mask = np.zeros((self.action_space.n,), dtype=bool)
        for spec in self.backend.legal_action_specs(self.state):
            idx = self._encode_action(spec)
            if 0 <= idx < self.action_space.n:
                mask[idx] = True
        return mask

    def _encode_action(self, spec: ActionSpec) -> int:
        if spec.kind == "skill":
            skill_idx = int(spec.skill_idx or 0)
            target_idx = int(spec.target_idx or 0)
            if not (0 <= skill_idx < MAX_SKILLS_PER_UNIT and 0 <= target_idx < ACTION_TARGET_SLOTS):
                return -1
            base = skill_idx * (ACTION_TARGET_SLOTS * 2)
            side_offset = ACTION_TARGET_SLOTS if spec.target_side == "heroes" else 0
            return base + side_offset + target_idx
        if spec.kind == "item":
            item_ids = list(self.state.items_available.keys())[:NUM_ITEMS]
            ii = item_ids.index(spec.item_id) if spec.item_id in item_ids else 0
            target_idx = int(spec.target_idx or 0)
            if not (0 <= target_idx < ACTION_TARGET_SLOTS):
                return -1
            side_offset = ACTION_TARGET_SLOTS if spec.target_side == "heroes" else 0
            return MAX_SKILLS_PER_UNIT * (ACTION_TARGET_SLOTS * 2) + ii * (ACTION_TARGET_SLOTS * 2) + side_offset + target_idx
        if spec.kind == "move":
            item_zone = MAX_SKILLS_PER_UNIT * (ACTION_TARGET_SLOTS * 2) + NUM_ITEMS * (ACTION_TARGET_SLOTS * 2)
            try:
                move_idx = ACTION_MOVE_DELTAS.index(int(spec.move_delta))
            except ValueError:
                return -1
            return item_zone + move_idx
        return ACTION_SPACE_SIZE - 1

    def _decode_action(self, action: int) -> ActionSpec:
        if action >= ACTION_SPACE_SIZE - 1:
            return ActionSpec(kind="pass")
        skill_zone = MAX_SKILLS_PER_UNIT * (ACTION_TARGET_SLOTS * 2)
        if action < skill_zone:
            width = ACTION_TARGET_SLOTS * 2
            skill_idx = action // width
            rem = action % width
            target_side = "heroes" if rem >= ACTION_TARGET_SLOTS else "enemies"
            target_idx = rem % ACTION_TARGET_SLOTS
            return ActionSpec(kind="skill", skill_idx=skill_idx, target_idx=target_idx, target_side=target_side)
        item_offset = action - skill_zone
        item_zone = NUM_ITEMS * (ACTION_TARGET_SLOTS * 2)
        if item_offset >= item_zone:
            move_idx = min(max(0, item_offset - item_zone), MOVE_ACTIONS - 1)
            return ActionSpec(kind="move", move_delta=ACTION_MOVE_DELTAS[move_idx])
        item_width = ACTION_TARGET_SLOTS * 2
        ii = item_offset // item_width
        rem = item_offset % item_width
        target_side = "heroes" if rem >= ACTION_TARGET_SLOTS else "enemies"
        ti = rem % ACTION_TARGET_SLOTS
        item_ids = list(self.state.items_available.keys())[:NUM_ITEMS]
        item_id = item_ids[ii] if ii < len(item_ids) else (item_ids[0] if item_ids else "")
        return ActionSpec(kind="item", item_id=item_id, target_idx=ti, target_side=target_side)

    def _obs(self, state: BattleState) -> np.ndarray:
        vec: list[float] = []
        token_order = list(TokenId)
        units = (state.heroes + state.enemies)[:8]
        for i in range(8):
            if i < len(units):
                u = units[i]
                tmap = {t.id: t.count for t in u.tokens}
                vec.extend([
                    max(0.0, u.hp / max(1, u.max_hp)),
                    min(1.0, u.stress / 10.0),
                    min(1.0, u.rank / 4.0),
                    min(1.0, getattr(u, "speed", 0) / 12.0),
                    min(1.0, getattr(u, "move_distance", 1) / 3.0),
                    min(1.0, getattr(u, "deathblow_resist", 0.0)),
                    1.0 if u.alive else 0.0,
                    1.0 if u.afflicted else 0.0,
                    1.0 if u.is_remnant else 0.0,
                ])
                vec.extend([min(1.0, tmap.get(tok, 0) / 3.0) for tok in token_order])
            else:
                vec.extend([0.0] * (9 + len(token_order)))
        vec.extend([min(1.0, state.round / 20.0), (state.turn_idx % 8) / 8.0])
        for iid in list(state.items_available.keys())[:NUM_ITEMS]:
            vec.append(min(1.0, state.items_available.get(iid, 0)))
        try:
            _, _, active = self.backend.get_active_unit(state)
            active_skills = self.backend._skills_for(active)[:MAX_SKILLS_PER_UNIT]
        except Exception:
            active = None
            active_skills = []
        for i in range(MAX_SKILLS_PER_UNIT):
            if active is not None and i < len(active_skills):
                sk = active_skills[i]
                if sk.charges > 0:
                    vec.append(min(1.0, state.skill_charges.get((active.id, sk.id), 0) / max(1, sk.charges)))
                else:
                    vec.append(1.0)
            else:
                vec.append(0.0)
        eid = (self.backend.current_encounter_id or "").lower()
        if "elite_swarm" in eid:
            vec.extend([1.0, 0.0, 0.0, 0.0])
        elif "elite_pair" in eid:
            vec.extend([0.0, 1.0, 0.0, 0.0])
        elif "holdout" in eid:
            vec.extend([0.0, 0.0, 1.0, 0.0])
        else:
            vec.extend([0.0, 0.0, 0.0, 1.0])
        # Fill the reserved auxiliary area with identity/tactical hints without
        # changing obs_dim: active hero identity plus compact active skill roles.
        aux: list[float] = []
        hero_order = ("man_at_arms", "hellion", "plague_doctor", "highwayman")
        active_arch = getattr(active, "archetype_id", "") if active is not None else ""
        aux.extend([1.0 if active_arch == hid else 0.0 for hid in hero_order])
        for sk in active_skills[:MAX_SKILLS_PER_UNIT]:
            aux.extend(
                [
                    1.0 if sk.damage_hi > 0 and not sk.is_friendly else 0.0,
                    1.0 if (sk.heal > 0 or sk.heal_percent > 0) else 0.0,
                    1.0 if sk.dot_type is not None and sk.dot_amount > 0 else 0.0,
                    1.0 if (sk.gives_target or sk.gives_self) else 0.0,
                ]
            )
        aux.append(1.0 if any(h.alive and (h.hp <= 1 or h.hp / max(1, h.max_hp) <= 0.25) for h in state.heroes) else 0.0)
        aux.append(1.0 if any(e.alive and e.hp <= max(2, int(e.max_hp * 0.15)) for e in state.enemies) else 0.0)
        consecutive_moves = int(getattr(self.reward_calculator, "_consecutive_moves", 0) or 0)
        last_skill_name = str(getattr(self.reward_calculator, "_last_skill_name", "") or "")
        aux.append(min(1.0, consecutive_moves / 4.0))
        aux.append(1.0 if last_skill_name == "move" else 0.0)
        for value in aux:
            if len(vec) >= self.observation_space.shape[0] - 16:
                break
            vec.append(float(value))
        while len(vec) < self.observation_space.shape[0] - 16:
            vec.append(0.0)
        rel_units = state.heroes[:4]
        for a in range(4):
            for b in range(4):
                if a < len(rel_units) and b < len(rel_units) and a != b:
                    key = (rel_units[a].id, rel_units[b].id)
                    vec.append(float(state.relationships.get(key, 0)) / 20.0)
                else:
                    vec.append(0.0)
        return np.asarray(vec, dtype=np.float32)

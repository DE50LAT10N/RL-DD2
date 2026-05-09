# Core turn-based DD2 battle simulator.
# Applies skills, tokens, DOTs, movement, initiative, deaths-door, and enemy behavior for RL training.
# Depends on env.game_data fixtures/overrides and emits ActionTrace for reward shaping.

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from env.data_model import ActionSpec, BattleState, Skill, Token, TokenId, Unit
from env.game_data import CombatItemSpec, GameData, load


@dataclass(slots=True)
class ActionTrace:
    actor_name: str = ""
    skill_name: str = ""
    target_names: list[str] = field(default_factory=list)
    damage_to_enemies: int = 0
    damage_to_heroes: int = 0
    healing_done: int = 0
    tokens_cured: int = 0
    combo_consumed: bool = False
    relationship_actout: str = ""
    extra_action: bool = False
    stress_delta: int = 0
    relationships_delta: int = 0
    used_item: str | None = None
    crit: bool = False
    legal_non_move_actions: int = 0
    legal_offensive_actions: int = 0


DOT_TOKENS = {TokenId.BURN, TokenId.BLEED, TokenId.BLIGHT, TokenId.POISON}
TOKEN_RESIST = {
    TokenId.BURN: "burn",
    TokenId.BLEED: "bleed",
    TokenId.BLIGHT: "blight",
    TokenId.POISON: "blight",
    TokenId.STUN: "stun",
    TokenId.DAZE: "stun",
    TokenId.IMMOBILIZE: "move",
    TokenId.WEAK: "debuff",
    TokenId.WINDED: "debuff",
    TokenId.VULNERABLE: "debuff",
    TokenId.BLIND: "debuff",
    TokenId.TAUNT: "debuff",
    TokenId.COMBO: "debuff",
}
INVERTS = {
    TokenId.STRENGTH: (TokenId.WEAK,),
    TokenId.WEAK: (TokenId.STRENGTH,),
    TokenId.SPEED: (TokenId.DAZE,),
    TokenId.DAZE: (TokenId.SPEED,),
    TokenId.STUN: (TokenId.SPEED,),
    TokenId.DODGE: (TokenId.BLIND,),
    TokenId.DODGE_PLUS: (TokenId.BLIND,),
    TokenId.BLIND: (TokenId.DODGE, TokenId.DODGE_PLUS),
    TokenId.BLOCK: (TokenId.VULNERABLE,),
    TokenId.BLOCK_PLUS: (TokenId.VULNERABLE,),
    TokenId.VULNERABLE: (TokenId.BLOCK, TokenId.BLOCK_PLUS),
    TokenId.TAUNT: (TokenId.STEALTH,),
    TokenId.STEALTH: (TokenId.TAUNT,),
}
TOKEN_LIMITS = {
    TokenId.CRIT: 2,
    TokenId.STRENGTH: 2,
    TokenId.WEAK: 2,
    TokenId.WINDED: 3,
    TokenId.VULNERABLE: 2,
    TokenId.BLIND: 2,
    TokenId.COMBO: 1,
    TokenId.SPEED: 1,
    TokenId.DAZE: 1,
    TokenId.STUN: 1,
    TokenId.IMMOBILIZE: 3,
    TokenId.DEATH_ARMOR: 3,
    TokenId.DEATHS_DOOR: 1,
}
TOKEN_DURATIONS = {
    TokenId.SPEED: 1,
    TokenId.DAZE: 1,
    TokenId.STUN: 1,
    TokenId.IMMOBILIZE: 1,
    TokenId.STEALTH: 1,
    TokenId.DEATH_ARMOR: 0,
    TokenId.DEATHS_DOOR: 0,
}


class SimBattleBackend:
    def __init__(self, seed: int = 7) -> None:
        self.data: GameData = load()
        self.rng = random.Random(seed)
        self.last_trace = ActionTrace()
        self.current_encounter_id: str | None = None
        self._item_cooldowns: dict[str, int] = {}

    def set_encounter_weights(self, weights: dict[str, float]) -> None:
        _ = weights

    def _build_unit(self, side: str, archetype_id: str, idx: int) -> Unit:
        if side == "heroes":
            archetype = self.data.heroes.get(archetype_id)
        else:
            archetype = self.data.monsters.get(archetype_id)
        hp = int(getattr(archetype, "hp", 36 if side == "heroes" else 24) or (36 if side == "heroes" else 24))
        move_distance = max(1, min(3, int(getattr(archetype, "move_distance", 1) or 1)))
        unit = Unit(
            id=f"{side[:1]}_{idx}_{archetype_id}",
            archetype_id=archetype_id,
            side="heroes" if side == "heroes" else "enemies",
            rank=idx + 1,
            hp=hp,
            max_hp=hp,
            move_distance=move_distance,
            speed=int(getattr(archetype, "speed", 5) or 5),
            size=max(1, min(4, int(getattr(archetype, "size", 1) or 1))),
            turns_per_round=max(1, int(getattr(archetype, "turns_per_round", 1) or 1)),
            resistances={str(k).lower(): float(v) for k, v in getattr(archetype, "resistances", {}).items()},
            deathblow_resist=float(getattr(archetype, "deathblow_resist", 0.6) or 0.6),
            stress=0,
        )
        death_armor = int(getattr(archetype, "death_armor", 0) or 0)
        if unit.side == "enemies" and death_armor > 0:
            self._add_token(unit, TokenId.DEATH_ARMOR, death_armor)
        return unit

    def _effective_speed(self, unit: Unit) -> int:
        speed = int(getattr(unit, "speed", 5) or 5)
        speed -= 3 * self._token_count(unit, TokenId.WINDED)
        if self._token_count(unit, TokenId.SPEED):
            speed += 100
        if self._token_count(unit, TokenId.DAZE):
            speed -= 100
        return speed

    def _rebuild_speed_order(self, state: BattleState) -> None:
        alive = [u for u in state.heroes + state.enemies if u.alive and not u.is_remnant]
        scored = []
        for u in alive:
            for turn_no in range(max(1, int(getattr(u, "turns_per_round", 1) or 1))):
                round_speed = self._effective_speed(u) + self.rng.randint(0, 6) - (2 * turn_no)
                scored.append((-round_speed, self.rng.random(), u.id))
        scored.sort()
        state.speed_order = [uid for _, _, uid in scored]
        state.initiative = list(state.speed_order)

    def _advance_turn(self, state: BattleState) -> None:
        state.item_used_actor_id = None
        state.turn_idx += 1
        while state.extra_action_queue:
            uid = state.extra_action_queue.pop(0)
            unit = self._get_unit_by_id(state, uid)
            if unit is not None and unit.alive and not unit.is_remnant:
                state.speed_order.insert(0, uid)
                return
        if not state.speed_order:
            state.round += 1
            self._end_round_decay(state)
            self._rebuild_speed_order(state)

    def reset(
        self,
        seed: int | None = None,
        encounter_id: str | None = None,
        hero_lineup: list[str] | None = None,
    ) -> BattleState:
        if seed is not None:
            self.rng.seed(seed)
        if encounter_id is None:
            encounter_keys = list(self.data.encounters.keys())
            chosen = self.rng.choice(encounter_keys) if encounter_keys else next(iter(self.data.encounters))
        else:
            chosen = encounter_id
        self.current_encounter_id = chosen
        enc = self.data.encounters.get(chosen)
        if enc is None:
            enc = next(iter(self.data.encounters.values()))
        if hero_lineup:
            hero_ids = [h.strip() for h in hero_lineup[:4] if str(h).strip()]
        else:
            hero_ids = list(enc.heroes or [])[:4] if enc.heroes else []
        if not hero_ids:
            hero_ids = list(self.data.heroes.keys())[:4]
        heroes = [self._build_unit("heroes", hid, i) for i, hid in enumerate(hero_ids)]
        enemies = [self._build_unit("enemies", mid, i) for i, mid in enumerate((enc.enemies or list(self.data.monsters))[:4])]
        state = BattleState(
            heroes=heroes,
            enemies=enemies,
            round=1,
            initiative=[],
            turn_idx=0,
            speed_order=[],
            skill_cooldowns={},
            skill_charges={},
            items_available={iid: 1 for iid in self.data.items.keys()},
            relationships={(a.id, b.id): 9 for a in heroes for b in heroes if a.id != b.id},
            extra_action_queue=[],
        )
        for unit in heroes + enemies:
            for sk in self._skills_for(unit):
                if sk.charges > 0:
                    state.skill_charges[(unit.id, sk.id)] = sk.charges
        self._rebuild_speed_order(state)
        self._item_cooldowns = {iid: 0 for iid in self.data.items.keys()}
        self.last_trace = ActionTrace()
        return state

    def _get_unit_by_id(self, state: BattleState, unit_id: str) -> Unit | None:
        for unit in state.heroes + state.enemies:
            if unit.id == unit_id:
                return unit
        return None

    def get_active_unit(self, state: BattleState) -> tuple[str, int, Unit]:
        while state.speed_order:
            uid = state.speed_order[0]
            unit = self._get_unit_by_id(state, uid)
            if unit is None or not unit.alive or unit.is_remnant:
                state.speed_order.pop(0)
                continue
            pool = state.heroes if unit.side == "heroes" else state.enemies
            return unit.side, pool.index(unit), unit
        self._rebuild_speed_order(state)
        return self.get_active_unit(state)

    @staticmethod
    def _token_count(unit: Unit, token: TokenId) -> int:
        return sum(t.count for t in unit.tokens if t.id == token)

    def _token_limit(self, token: TokenId) -> int:
        spec = self.data.tokens.get(token.value)
        return int(getattr(spec, "max_stacks", 0) or TOKEN_LIMITS.get(token, 3))

    def _token_duration(self, token: TokenId, duration: int | None = None) -> int:
        if duration is not None:
            return max(0, int(duration))
        spec = self.data.tokens.get(token.value)
        configured = int(getattr(spec, "default_duration", 0) or 0)
        if configured > 0:
            return configured
        return TOKEN_DURATIONS.get(token, 3)

    def _add_token(self, unit: Unit, token: TokenId, count: int = 1, *, duration: int | None = None, amount: int | None = None) -> None:
        remaining = max(0, int(count))
        if remaining <= 0:
            return
        for inverse in INVERTS.get(token, ()):
            while remaining > 0 and self._consume_once(unit, inverse):
                remaining -= 1
            if remaining <= 0:
                return
        token_duration = self._token_duration(token, duration)
        if token in DOT_TOKENS:
            dot_amount = max(1, int(amount if amount is not None else remaining))
            for t in unit.tokens:
                if t.id == token:
                    t.amount += dot_amount
                    t.count = t.amount
                    t.duration = max(t.duration, token_duration)
                    return
            unit.tokens.append(Token(id=token, count=dot_amount, duration=token_duration, amount=dot_amount))
            return
        limit = self._token_limit(token)
        for t in unit.tokens:
            if t.id == token:
                t.count = min(limit, t.count + remaining)
                t.duration = max(t.duration, token_duration)
                return
        unit.tokens.append(Token(id=token, count=min(limit, remaining), duration=token_duration))

    @staticmethod
    def _consume_once(unit: Unit, token: TokenId) -> bool:
        for t in list(unit.tokens):
            if t.id == token and t.count > 0:
                t.count -= 1
                if t.count <= 0:
                    unit.tokens.remove(t)
                return True
        return False

    @staticmethod
    def _remove_token(unit: Unit, token: TokenId) -> None:
        unit.tokens = [t for t in unit.tokens if t.id != token]

    @staticmethod
    def _cure_tokens(unit: Unit, tokens: list[TokenId]) -> int:
        if not tokens:
            return 0
        wanted = set(tokens)
        before = len(unit.tokens)
        unit.tokens = [t for t in unit.tokens if t.id not in wanted]
        return before - len(unit.tokens)

    @staticmethod
    def _has_any_token(unit: Unit, tokens: list[TokenId]) -> bool:
        wanted = set(tokens)
        return any(t.id in wanted for t in unit.tokens)

    @staticmethod
    def _heal_amount(skill: Skill, target: Unit) -> int:
        flat = max(0, int(skill.heal or 0))
        percent = max(0, int(math.ceil(target.max_hp * float(skill.heal_percent or 0.0))))
        return max(flat, percent)

    def _friendly_skill_can_affect(self, skill: Skill, target: Unit) -> bool:
        if not skill.is_friendly:
            return True
        if skill.heal_threshold > 0 and not self._has_any_token(target, skill.cures_tokens):
            if (target.hp / max(1, target.max_hp)) > skill.heal_threshold:
                return False
        return True

    @staticmethod
    def _friendly_skill_can_heal(skill: Skill, target: Unit) -> bool:
        if not skill.is_friendly or skill.heal_threshold <= 0:
            return True
        return (target.hp / max(1, target.max_hp)) <= skill.heal_threshold

    def _decay_turn_tokens(self, unit: Unit) -> None:
        for t in list(unit.tokens):
            if t.duration <= 0 or t.id == TokenId.DEATHS_DOOR:
                continue
            t.duration -= 1
            if t.duration <= 0:
                unit.tokens.remove(t)

    def _resist_chance(self, unit: Unit, key: str, pierce: float = 0.0) -> float:
        raw = float(getattr(unit, "resistances", {}).get(key.lower(), 0.0))
        if raw > 1.0:
            raw /= 100.0
        return max(0.0, min(0.95, raw - pierce))

    def _try_apply_token(self, unit: Unit, token: TokenId, count: int = 1, *, duration: int | None = None, amount: int | None = None, pierce: float = 0.0) -> bool:
        resist_key = TOKEN_RESIST.get(token)
        if resist_key and self.rng.random() < self._resist_chance(unit, resist_key, pierce):
            return False
        self._add_token(unit, token, count, duration=duration, amount=amount)
        return True

    def _consume_token_priority(self, unit: Unit, tokens: tuple[TokenId, ...]) -> TokenId | None:
        for token in tokens:
            if self._consume_once(unit, token):
                return token
        return None

    def _targetable_units(self, units: list[Unit], ranks: set[int], *, friendly: bool) -> list[tuple[int, Unit]]:
        candidates = [
            (idx, unit)
            for idx, unit in enumerate(units)
            if unit.alive and self._target_rankable(unit, ranks)
        ]
        if friendly:
            return candidates
        visible = [(idx, unit) for idx, unit in candidates if self._token_count(unit, TokenId.STEALTH) <= 0]
        return visible or candidates

    def _taunt_filtered_targets(self, candidates: list[tuple[int, Unit]], *, friendly: bool) -> list[tuple[int, Unit]]:
        if friendly:
            return candidates
        taunted = [(idx, unit) for idx, unit in candidates if self._token_count(unit, TokenId.TAUNT) > 0]
        return taunted or candidates

    def _apply_start_turn_dot(self, unit: Unit) -> None:
        for t in list(unit.tokens):
            if t.id in DOT_TOKENS and t.duration > 0:
                unit.hp -= max(1, int(t.amount or t.count))
                t.duration -= 1
                if t.duration <= 0:
                    unit.tokens.remove(t)
        for t in list(unit.tokens):
            if t.id == TokenId.REGENERATION and t.duration > 0:
                unit.hp = min(unit.max_hp, unit.hp + max(1, int(t.amount or t.count)))
                t.duration -= 1
                if t.duration <= 0:
                    unit.tokens.remove(t)
        if unit.hp <= 0:
            self._handle_deaths_door(unit, execution=0)

    def _handle_deaths_door(self, unit: Unit, *, execution: int = 0) -> None:
        if unit.is_remnant:
            unit.alive = False
            unit.hp = 0
            return
        if unit.side == "enemies":
            armor = self._token_count(unit, TokenId.DEATH_ARMOR)
            if execution >= armor and armor > 0:
                self._remove_token(unit, TokenId.DEATH_ARMOR)
                armor = 0
            if armor <= 0 and self._token_count(unit, TokenId.DEATHS_DOOR) <= 0:
                unit.alive = False
                unit.hp = 0
                return
            if armor > 0:
                for _ in range(max(1, execution or 1)):
                    if not self._consume_once(unit, TokenId.DEATH_ARMOR):
                        break
                unit.hp = 1
                self._add_token(unit, TokenId.DEATHS_DOOR, 1)
                return
            unit.alive = False
            unit.hp = 0
            return
        if self._token_count(unit, TokenId.DEATHS_DOOR) > 0:
            resist = max(0.0, min(0.9, unit.deathblow_resist - unit.deathblow_resist_penalty))
            if execution > 0 or self.rng.random() >= resist:
                unit.alive = False
                unit.hp = 0
                return
            unit.deathblow_resist_penalty += 0.1
            unit.hp = 1
            return
        unit.hp = 1
        self._add_token(unit, TokenId.DEATHS_DOOR, 1)
        self._add_token(unit, TokenId.WEAK, 2)
        self._add_token(unit, TokenId.DAZE, 1)

    def _make_remnant(self, unit: Unit) -> None:
        remnant_hp = max(1, min(12, int(round(unit.max_hp * 0.3))))
        unit.id = f"{unit.id}_tombstone"
        unit.archetype_id = "enemy_tombstone"
        unit.hp = remnant_hp
        unit.max_hp = remnant_hp
        unit.stress = 0
        unit.tokens = []
        unit.afflicted = False
        unit.alive = True
        unit.is_remnant = True

    def _roll_hit(self, actor: Unit, target: Unit) -> bool:
        blind_penalty = 0.5 if self._token_count(actor, TokenId.BLIND) else 0.0
        dodge_bonus = 0.75 if self._token_count(target, TokenId.DODGE_PLUS) else (0.5 if self._token_count(target, TokenId.DODGE) else 0.0)
        hit = self.rng.random() >= max(0.0, min(0.95, blind_penalty + dodge_bonus))
        self._consume_token_priority(actor, (TokenId.BLIND,))
        self._consume_token_priority(target, (TokenId.DODGE_PLUS, TokenId.DODGE))
        return hit

    def _resolve_guarded(self, state: BattleState, target: Unit) -> Unit:
        if self._token_count(target, TokenId.IMMOBILIZE):
            return target
        team = state.heroes if target.side == "heroes" else state.enemies
        guards = [u for u in team if u.alive and u.id != target.id and self._token_count(u, TokenId.TAUNT) > 0]
        if guards and self._token_count(target, TokenId.GUARDED):
            return guards[0]
        return target

    def _apply_damage(self, actor: Unit, target: Unit, skill: Skill) -> int:
        combo_active = self._token_count(target, TokenId.COMBO) > 0
        forced_crit = self._token_count(actor, TokenId.CRIT) > 0
        did_crit = forced_crit or self.rng.random() < skill.crit_chance
        base = math.ceil(skill.damage_hi * 1.5) if did_crit else self.rng.randint(skill.damage_lo, max(skill.damage_lo, skill.damage_hi))
        mult = 1.0
        if combo_active and skill.combo_damage_multiplier > 0:
            mult *= float(skill.combo_damage_multiplier)
        if self._token_count(actor, TokenId.STRENGTH):
            mult *= 1.5
            self._consume_once(actor, TokenId.STRENGTH)
        if self._token_count(actor, TokenId.WEAK):
            mult *= 0.5
            self._consume_once(actor, TokenId.WEAK)
        winded = self._token_count(actor, TokenId.WINDED)
        if winded > 0:
            mult *= max(0.2, 1.0 - 0.33 * winded)
        if self._token_count(target, TokenId.VULNERABLE):
            mult *= 1.5
            self._consume_once(target, TokenId.VULNERABLE)
        consumed_block = self._consume_token_priority(target, (TokenId.BLOCK_PLUS, TokenId.BLOCK))
        if consumed_block == TokenId.BLOCK_PLUS:
            mult *= 0.25
        elif consumed_block == TokenId.BLOCK:
            mult *= 0.5
        if forced_crit:
            self._consume_once(actor, TokenId.CRIT)
        if combo_active and skill.combo_consumes:
            self.last_trace.combo_consumed = self._consume_once(target, TokenId.COMBO)
        dmg = max(0, int(round(base * mult)))
        target.hp -= dmg
        self.last_trace.crit = did_crit
        if did_crit and actor.side == "heroes":
            actor.stress = max(0, actor.stress - 1)
        elif did_crit and target.side == "heroes":
            target.stress = min(10, target.stress + 1)
        if target.hp <= 0:
            self._handle_deaths_door(target, execution=skill.execution)
        return dmg

    def _rank_shift(self, side_units: list[Unit], unit: Unit, delta: int) -> None:
        if delta == 0 or not unit.alive or self._token_count(unit, TokenId.IMMOBILIZE):
            return
        alive = sorted([u for u in side_units if u.alive], key=lambda x: x.rank)
        if unit not in alive:
            return
        current_idx = alive.index(unit)
        target_idx = max(0, min(len(alive) - 1, current_idx + delta))
        if target_idx == current_idx:
            return
        alive.pop(current_idx)
        alive.insert(target_idx, unit)
        for i, u in enumerate(alive):
            u.rank = i + 1

    @staticmethod
    def _combat_alive(unit: Unit) -> bool:
        return unit.alive and not unit.is_remnant

    @staticmethod
    def _combat_units(units: list[Unit]) -> list[Unit]:
        return [u for u in units if u.alive and not u.is_remnant]

    @staticmethod
    def _remnant_units(units: list[Unit]) -> list[Unit]:
        return [u for u in units if u.alive and u.is_remnant]

    @staticmethod
    def _occupied_ranks(unit: Unit) -> set[int]:
        return set(range(int(unit.rank), min(4, int(unit.rank) + max(1, int(getattr(unit, "size", 1) or 1)) - 1) + 1))

    @classmethod
    def _target_rankable(cls, unit: Unit, ranks: set[int]) -> bool:
        return bool(cls._occupied_ranks(unit) & set(ranks))

    def _mark_terminal_if_done(self, state: BattleState) -> bool:
        if not any(u.alive for u in state.enemies):
            state.done = True
            state.heroes_won = True
            return True
        if not any(self._combat_alive(u) for u in state.heroes):
            state.done = True
            state.heroes_won = False
            return True
        return False

    def _apply_relationship_event(self, state: BattleState, actor: Unit, event_delta: int) -> None:
        for ally in state.heroes:
            if ally.id == actor.id:
                continue
            key = (actor.id, ally.id)
            state.relationships[key] = max(0, min(20, state.relationships.get(key, 0) + event_delta))

    def _relationship_actout_before_action(self, state: BattleState, actor: Unit, skill: Skill) -> None:
        if actor.side != "heroes" or not skill.id:
            return
        allies = [u for u in state.heroes if u.alive and u.id != actor.id]
        if not allies:
            return
        best_rel = max(state.relationships.get((ally.id, actor.id), 9) for ally in allies)
        worst_rel = min(state.relationships.get((ally.id, actor.id), 9) for ally in allies)
        if best_rel >= 14 and self.rng.random() < min(0.25, 0.04 + (best_rel - 13) * 0.025):
            self._add_token(actor, TokenId.STRENGTH if not skill.is_friendly else TokenId.BLOCK, 1)
            actor.stress = max(0, actor.stress - 1)
            self.last_trace.relationship_actout = "positive"
            self.last_trace.stress_delta -= 1
            return
        if worst_rel <= 5 and self.rng.random() < min(0.3, 0.06 + (6 - worst_rel) * 0.035):
            self._add_token(actor, TokenId.WEAK if not skill.is_friendly else TokenId.BLIND, 1)
            actor.stress = min(10, actor.stress + 1)
            self.last_trace.relationship_actout = "negative"
            self.last_trace.stress_delta += 1

    def _end_round_decay(self, state: BattleState) -> None:
        for key, value in list(state.skill_cooldowns.items()):
            if value > 0:
                state.skill_cooldowns[key] = value - 1
            if state.skill_cooldowns[key] <= 0:
                state.skill_cooldowns.pop(key, None)
        for iid in list(self._item_cooldowns):
            if self._item_cooldowns[iid] > 0:
                self._item_cooldowns[iid] -= 1

    def _skills_for(self, unit: Unit) -> list[Skill]:
        spec = self.data.heroes.get(unit.archetype_id)
        if spec is None:
            mspec = self.data.monsters.get(unit.archetype_id)
            raw_skills = mspec.skills if mspec else []
        else:
            raw_skills = spec.skills
        skills: list[Skill] = []
        for rs in raw_skills[:5]:
            dot_type = TokenId[rs.dot_type] if rs.dot_type in TokenId.__members__ else None
            skills.append(Skill(
                id=rs.id,
                source_ranks=set(rs.source_ranks),
                target_ranks=set(rs.target_ranks),
                cooldown=rs.cooldown,
                charges=rs.charges,
                is_friendly=rs.is_friendly,
                targets_self_party=rs.targets_self_party,
                target_self=rs.target_self,
                multi_target=rs.multi_target,
                move_self=rs.move_self,
                move_target=rs.move_target,
                damage_lo=rs.damage_lo,
                damage_hi=rs.damage_hi,
                crit_chance=rs.crit_chance,
                stress_damage=rs.stress_damage,
                heal=rs.heal,
                heal_percent=rs.heal_percent,
                heal_threshold=rs.heal_threshold,
                heal_stress=rs.heal_stress,
                cures_tokens=[TokenId[t] for t in rs.cures_tokens if t in TokenId.__members__],
                costs=[Token(TokenId[k], v) for k, v in rs.costs.items() if k in TokenId.__members__],
                gives_self=[Token(TokenId[k], v) for k, v in rs.gives_self.items() if k in TokenId.__members__],
                gives_target=[Token(TokenId[k], v) for k, v in rs.gives_target.items() if k in TokenId.__members__],
                combo_gives_self=[Token(TokenId[k], v) for k, v in rs.combo_gives_self.items() if k in TokenId.__members__],
                combo_gives_target=[Token(TokenId[k], v) for k, v in rs.combo_gives_target.items() if k in TokenId.__members__],
                dot_type=dot_type,
                dot_amount=rs.dot_amount,
                dot_duration=rs.dot_duration,
                combo_damage_multiplier=rs.combo_damage_multiplier,
                combo_dot_amount=rs.combo_dot_amount,
                combo_dot_duration=rs.combo_dot_duration,
                combo_consumes=rs.combo_consumes,
                extra_action_self=rs.extra_action_self,
                pierce={str(k).lower(): float(v) for k, v in rs.pierce.items()},
                execution=rs.execution,
            ))
        if not skills:
            skills.append(Skill(id="strike", source_ranks={1, 2, 3, 4}, target_ranks={1, 2, 3, 4}, damage_lo=3, damage_hi=6))
        return skills

    def legal_action_specs(self, state: BattleState) -> list[ActionSpec]:
        if state.done:
            return [ActionSpec(kind="pass")]
        side, actor_idx, actor = self.get_active_unit(state)
        if side != "heroes" or not actor.alive:
            return [ActionSpec(kind="pass")]
        actions: list[ActionSpec] = []
        skills = self._skills_for(actor)
        for si, sk in enumerate(skills[:5]):
            if state.skill_cooldowns.get((actor.id, sk.id), 0) > 0:
                continue
            if sk.charges > 0 and state.skill_charges.get((actor.id, sk.id), 0) <= 0:
                continue
            if actor.rank not in sk.source_ranks:
                continue
            if any(self._token_count(actor, cost.id) < cost.count for cost in sk.costs):
                continue
            targets = state.heroes if sk.is_friendly else state.enemies
            target_side = "heroes" if sk.is_friendly else "enemies"
            candidate_targets = self._targetable_units(targets, sk.target_ranks, friendly=sk.is_friendly)
            candidate_targets = self._taunt_filtered_targets(candidate_targets, friendly=sk.is_friendly)
            if sk.is_friendly:
                candidate_targets = [
                    (ti, tgt)
                    for ti, tgt in candidate_targets
                    if self._friendly_skill_can_affect(sk, tgt)
                ]
            if sk.target_self:
                candidate_targets = [(state.heroes.index(actor), actor)] if actor in state.heroes and sk.is_friendly else []
                candidate_targets = [
                    (ti, tgt)
                    for ti, tgt in candidate_targets
                    if self._friendly_skill_can_affect(sk, tgt)
                ]
            for ti, tgt in candidate_targets:
                actions.append(ActionSpec(kind="skill", actor_idx=actor_idx, skill_idx=si, target_idx=ti, target_side=target_side))
        if state.item_used_actor_id != actor.id:
            for iid, amount in state.items_available.items():
                if amount <= 0 or self._item_cooldowns.get(iid, 0) > 0:
                    continue
                item = self.data.items[iid]
                target_side = "enemies" if item.target_side == "enemies" else "heroes"
                targets = state.enemies if target_side == "enemies" else state.heroes
                for ti, tgt in enumerate(targets):
                    if tgt.alive:
                        actions.append(ActionSpec(kind="item", actor_idx=actor_idx, item_id=iid, target_idx=ti, target_side=target_side))
        if not self._token_count(actor, TokenId.IMMOBILIZE):
            max_move = max(1, min(3, int(getattr(actor, "move_distance", 1) or 1)))
            team = state.heroes if actor.side == "heroes" else state.enemies
            alive_count = len([u for u in team if u.alive])
            for delta in range(-max_move, max_move + 1):
                if delta == 0:
                    continue
                target_rank = actor.rank + delta
                max_rank = max(1, 5 - max(1, int(getattr(actor, "size", 1) or 1)))
                if 1 <= target_rank <= min(alive_count, max_rank):
                    actions.append(ActionSpec(kind="move", actor_idx=actor_idx, move_delta=delta))
        if actions:
            return actions
        return [ActionSpec(kind="pass", actor_idx=actor_idx, target_idx=None)]

    def get_valid_actions(self, state: BattleState) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for a in self.legal_action_specs(state):
            if a.kind == "skill" and a.skill_idx is not None and a.target_idx is not None:
                pairs.append((a.skill_idx, a.target_idx))
        return pairs

    def _apply_item(self, state: BattleState, item: CombatItemSpec, actor: Unit, target: Unit) -> None:
        hp_before = target.hp
        target.hp = min(target.max_hp, target.hp + item.heal)
        actual_heal = max(0, target.hp - hp_before)
        if target.hp > 1:
            self._remove_token(target, TokenId.DEATHS_DOOR)
            target.deathblow_resist_penalty = 0.0
        target.stress = max(0, target.stress - item.stress_heal)
        for tk, cnt in item.gives_target.items():
            if tk in TokenId.__members__:
                self._try_apply_token(target, TokenId[tk], cnt)
        state.items_available[item.id] = max(0, state.items_available.get(item.id, 0) - 1)
        self._item_cooldowns[item.id] = item.cooldown
        state.item_used_actor_id = actor.id
        self.last_trace = ActionTrace(actor_name=actor.id, skill_name="item", target_names=[target.id], healing_done=actual_heal, used_item=item.id)

    def _pick_enemy_action(self, state: BattleState, actor: Unit) -> ActionSpec:
        skills = [
            s for s in self._skills_for(actor)
            if actor.rank in s.source_ranks
            and state.skill_cooldowns.get((actor.id, s.id), 0) <= 0
            and (s.charges <= 0 or state.skill_charges.get((actor.id, s.id), 0) > 0)
        ]
        heroes = [h for h in state.heroes if self._combat_alive(h)]
        if not skills or not heroes:
            delta = self._choose_reposition_delta(state, actor)
            return ActionSpec(kind="move", move_delta=delta) if delta else ActionSpec(kind="pass")
        taunt_targets = [h for h in heroes if self._token_count(h, TokenId.TAUNT) > 0]
        if taunt_targets:
            target = taunt_targets[0]
        else:
            target = min(heroes, key=lambda h: (h.hp / max(1, h.max_hp), -h.stress))
        target_idx = state.heroes.index(target)
        targetable_skills = [
            s for s in skills
            if self._target_rankable(target, s.target_ranks)
            and (self._token_count(target, TokenId.STEALTH) <= 0 or not any(self._token_count(h, TokenId.STEALTH) <= 0 for h in heroes if h.id != target.id))
        ]
        if not targetable_skills:
            delta = self._choose_reposition_delta(state, actor)
            return ActionSpec(kind="move", move_delta=delta) if delta else ActionSpec(kind="pass")
        best = max(targetable_skills, key=lambda s: (s.damage_hi, s.stress_damage))
        skill_idx = self._skills_for(actor).index(best)
        return ActionSpec(kind="skill", skill_idx=skill_idx, target_idx=target_idx, target_side="heroes")

    def _has_attack_targets(self, state: BattleState, actor: Unit) -> bool:
        """True when actor can use at least one offensive skill right now."""
        for sk in self._skills_for(actor):
            if sk.is_friendly:
                continue
            if actor.rank not in sk.source_ranks:
                continue
            if state.skill_cooldowns.get((actor.id, sk.id), 0) > 0:
                continue
            if sk.charges > 0 and state.skill_charges.get((actor.id, sk.id), 0) <= 0:
                continue
            for tgt in state.enemies if actor.side == "heroes" else state.heroes:
                if tgt.alive and self._target_rankable(tgt, sk.target_ranks):
                    return True
        return False

    def _attack_score_at_rank(self, state: BattleState, actor: Unit, rank: int) -> float:
        """Estimate how strong actor's attack options are from a given rank."""
        score = 0.0
        targets = state.enemies if actor.side == "heroes" else state.heroes
        for sk in self._skills_for(actor):
            if sk.is_friendly:
                continue
            if rank not in sk.source_ranks:
                continue
            if state.skill_cooldowns.get((actor.id, sk.id), 0) > 0:
                continue
            if sk.charges > 0 and state.skill_charges.get((actor.id, sk.id), 0) <= 0:
                continue
            valid_targets = sum(1 for tgt in targets if tgt.alive and self._target_rankable(tgt, sk.target_ranks))
            if valid_targets <= 0:
                continue
            # Favor positions with more reachable targets and stronger damage ceilings.
            score += valid_targets + (0.1 * max(0, sk.damage_hi))
        return score

    def _choose_reposition_delta(self, state: BattleState, actor: Unit) -> int:
        """Choose one-step rank shift that improves attack options the most."""
        current_rank = int(actor.rank)
        candidates: list[tuple[int, float]] = []
        move_distance = max(1, min(3, int(getattr(actor, "move_distance", 1) or 1)))
        for target_rank in range(current_rank - move_distance, current_rank + move_distance + 1):
            if target_rank == current_rank:
                continue
            if not (1 <= target_rank <= 4):
                continue
            candidates.append((target_rank, self._attack_score_at_rank(state, actor, target_rank)))
        if not candidates:
            return 0
        best_rank, best_score = max(candidates, key=lambda x: (x[1], -abs(x[0] - 2.5)))
        current_score = self._attack_score_at_rank(state, actor, current_rank)
        if best_score <= current_score and self._has_attack_targets(state, actor):
            return 0
        return best_rank - current_rank

    def _resolve_stress_event(self, state: BattleState, unit: Unit) -> None:
        if unit.stress < 10 or not unit.alive:
            return
        unit.stress = 0
        if self.rng.random() < 0.2:
            unit.afflicted = False
            unit.hp = min(unit.max_hp, max(unit.hp, int(math.ceil(unit.max_hp * 0.5))))
            self._add_token(unit, TokenId.STRENGTH, 1)
            self._add_token(unit, TokenId.BLOCK, 1)
            if unit.side == "heroes":
                self._apply_relationship_event(state, unit, 1)
            return
        unit.afflicted = True
        unit.hp = min(unit.hp, max(1, int(math.ceil(unit.max_hp * 0.1))))
        self._add_token(unit, TokenId.WEAK, 1)
        self._add_token(unit, TokenId.VULNERABLE, 1)
        if unit.side == "heroes":
            self._apply_relationship_event(state, unit, -3)

    def execute_spec(self, state: BattleState, chosen: ActionSpec) -> BattleState:
        if state.done:
            return state
        side, actor_idx, actor = self.get_active_unit(state)
        first_action_this_turn = state.item_used_actor_id != actor.id
        if first_action_this_turn:
            self._apply_start_turn_dot(actor)
            if self._mark_terminal_if_done(state):
                return state
        if not actor.alive or actor.is_remnant:
            if state.speed_order:
                state.speed_order.pop(0)
            self._advance_turn(state)
            return state
        self.last_trace = ActionTrace(actor_name=actor.id)

        if self._token_count(actor, TokenId.STUN):
            self.last_trace.skill_name = "stunned"
            self._consume_once(actor, TokenId.STUN)
            if state.speed_order:
                state.speed_order.pop(0)
            self._advance_turn(state)
            return state

        if side == "heroes":
            legal = self.legal_action_specs(state)
            self.last_trace.legal_non_move_actions = sum(1 for a in legal if a.kind != "move")
            self.last_trace.legal_offensive_actions = self._count_offensive_legal_actions(state, actor, legal)
            if chosen not in legal:
                self.last_trace.skill_name = "invalid_action"
                if state.speed_order:
                    state.speed_order.pop(0)
                self._advance_turn(state)
                return state
        else:
            chosen = self._pick_enemy_action(state, actor)

        if chosen.kind == "pass":
            self.last_trace.skill_name = "pass"
        elif chosen.kind == "move":
            self.last_trace.skill_name = "move"
            team = state.heroes if actor.side == "heroes" else state.enemies
            self._rank_shift(team, actor, int(chosen.move_delta))
        elif chosen.kind == "item" and chosen.item_id:
            item = self.data.items[chosen.item_id]
            target_team = state.enemies if item.target_side == "enemies" else state.heroes
            target = target_team[min(chosen.target_idx or 0, len(target_team) - 1)]
            self._apply_item(state, item, actor, target)
            self._mark_terminal_if_done(state)
            return state
        else:
            skills = self._skills_for(actor)
            sk = skills[min(chosen.skill_idx or 0, len(skills) - 1)]
            target_team = state.heroes if (chosen.target_side == "heroes" or sk.is_friendly) else state.enemies
            if not any(u.alive for u in target_team):
                self.last_trace.skill_name = "pass_no_targets"
            else:
                target = target_team[min(chosen.target_idx or 0, len(target_team) - 1)]
                if not target.alive:
                    self.last_trace.skill_name = "pass_no_targets"
                elif target.is_remnant and (sk.is_friendly or target.side != "enemies"):
                    self.last_trace.skill_name = "pass_no_targets"
                else:
                    self._relationship_actout_before_action(state, actor, sk)
                    target = self._resolve_guarded(state, target)
                    self.last_trace.skill_name = sk.id
                    self.last_trace.target_names = [target.id]
                    targets_to_apply = [target]
                    if sk.multi_target:
                        targets_to_apply = [
                            u for u in target_team
                            if u.alive and self._target_rankable(u, sk.target_ranks)
                        ]
                        if sk.is_friendly:
                            targets_to_apply = [
                                u for u in targets_to_apply
                                if self._friendly_skill_can_affect(sk, u)
                            ]
                        self.last_trace.target_names = [u.id for u in targets_to_apply]
                    for cost in sk.costs:
                        for _ in range(cost.count):
                            self._consume_once(actor, cost.id)
                    combo_seen_this_action = False
                    for actual_target in targets_to_apply:
                        combo_active_before_effects = self._token_count(actual_target, TokenId.COMBO) > 0
                        combo_seen_this_action = combo_seen_this_action or combo_active_before_effects
                        if sk.is_friendly or self._roll_hit(actor, actual_target):
                            if sk.damage_hi > 0 and not sk.is_friendly:
                                dmg = self._apply_damage(actor, actual_target, sk)
                                if actual_target.side == "heroes":
                                    self.last_trace.damage_to_heroes += dmg
                                else:
                                    self.last_trace.damage_to_enemies += dmg
                            heal_amount = self._heal_amount(sk, actual_target)
                            if heal_amount > 0 and self._friendly_skill_can_heal(sk, actual_target):
                                hp_before = actual_target.hp
                                actual_target.hp = min(actual_target.max_hp, actual_target.hp + heal_amount)
                                if actual_target.hp > 1:
                                    self._remove_token(actual_target, TokenId.DEATHS_DOOR)
                                    actual_target.deathblow_resist_penalty = 0.0
                                self.last_trace.healing_done += max(0, actual_target.hp - hp_before)
                            if sk.cures_tokens:
                                self.last_trace.tokens_cured += self._cure_tokens(actual_target, sk.cures_tokens)
                            if sk.heal_stress > 0:
                                old_stress = actual_target.stress
                                actual_target.stress = max(0, actual_target.stress - sk.heal_stress)
                                self.last_trace.stress_delta += actual_target.stress - old_stress
                            old_stress = actual_target.stress
                            actual_target.stress = min(10, actual_target.stress + sk.stress_damage)
                            self.last_trace.stress_delta += actual_target.stress - old_stress
                            for t in sk.gives_target:
                                self._try_apply_token(actual_target, t.id, t.count, pierce=sk.pierce.get(TOKEN_RESIST.get(t.id, ""), 0.0))
                            if combo_active_before_effects:
                                for t in sk.combo_gives_target:
                                    self._try_apply_token(actual_target, t.id, t.count, pierce=sk.pierce.get(TOKEN_RESIST.get(t.id, ""), 0.0))
                            if sk.dot_type and sk.dot_amount > 0:
                                dot_amount = sk.dot_amount
                                dot_duration = sk.dot_duration
                                if combo_active_before_effects and sk.combo_dot_amount > 0:
                                    dot_amount += sk.combo_dot_amount
                                    dot_duration = max(dot_duration, sk.combo_dot_duration or dot_duration)
                                self._try_apply_token(
                                    actual_target,
                                    sk.dot_type,
                                    dot_amount,
                                    duration=dot_duration,
                                    amount=dot_amount,
                                    pierce=sk.pierce.get(TOKEN_RESIST.get(sk.dot_type, ""), 0.0),
                                )
                            if self._token_count(actual_target, TokenId.RIPOSTE) and actual_target.side != actor.side:
                                self._consume_once(actual_target, TokenId.RIPOSTE)
                                back = max(1, self.rng.randint(1, 4))
                                actor.hp -= back
                                if actor.side == "heroes":
                                    self.last_trace.damage_to_heroes += back
                                else:
                                    self.last_trace.damage_to_enemies += back
                                if actor.hp <= 0:
                                    self._handle_deaths_door(actor, execution=0)
                            self._rank_shift(state.heroes if actual_target.side == "heroes" else state.enemies, actual_target, sk.move_target)
                    for t in sk.gives_self:
                        self._add_token(actor, t.id, t.count)
                    if combo_seen_this_action:
                        for t in sk.combo_gives_self:
                            self._add_token(actor, t.id, t.count)
                    if sk.cooldown > 0:
                        state.skill_cooldowns[(actor.id, sk.id)] = sk.cooldown
                    if sk.charges > 0:
                        key = (actor.id, sk.id)
                        state.skill_charges[key] = max(0, state.skill_charges.get(key, sk.charges) - 1)
                    if sk.extra_action_self:
                        state.extra_action_queue.append(actor.id)
                        self.last_trace.extra_action = True
                    self._rank_shift(state.heroes if actor.side == "heroes" else state.enemies, actor, sk.move_self)
                    rel_delta = 1 if sk.heal > 0 else (-1 if sk.stress_damage > 0 else 0)
                    if rel_delta and actor.side == "heroes":
                        self._apply_relationship_event(state, actor, rel_delta)
                        self.last_trace.relationships_delta += rel_delta

        for u in state.heroes + state.enemies:
            self._resolve_stress_event(state, u)

        self._mark_terminal_if_done(state)

        if state.speed_order:
            state.speed_order.pop(0)
        self._decay_turn_tokens(actor)
        self._advance_turn(state)
        return state

    def _count_offensive_legal_actions(self, state: BattleState, actor: Unit, legal: list[ActionSpec]) -> int:
        skills = self._skills_for(actor)
        count = 0
        for action in legal:
            if action.kind == "item":
                item = self.data.items.get(str(action.item_id or ""))
                if item is not None and item.target_side == "enemies" and (
                    int(getattr(item, "damage", 0) or 0) > 0
                    or int(getattr(item, "dot_amount", 0) or 0) > 0
                    or bool(getattr(item, "gives_target", {}) or {})
                ):
                    count += 1
                continue
            if action.kind != "skill" or action.target_side != "enemies" or action.skill_idx is None:
                continue
            if not (0 <= int(action.skill_idx) < len(skills)):
                continue
            skill = skills[int(action.skill_idx)]
            if skill.damage_hi > 0 or skill.dot_amount > 0 or skill.execution > 0 or skill.stress_damage > 0:
                count += 1
        return count

    def execute_action(self, state: BattleState, skill_idx: int, target_idx: int) -> BattleState:
        # Test/helper path: force a hero action by moving an alive hero to the head
        # of the queue when the current actor is an enemy.
        side, _, _ = self.get_active_unit(state)
        if side != "heroes":
            hero = next((h for h in state.heroes if h.alive), None)
            if hero is not None:
                state.speed_order = [hero.id] + [uid for uid in state.speed_order if uid != hero.id]
        return self.execute_spec(
            state,
            ActionSpec(kind="skill", actor_idx=0, skill_idx=skill_idx, target_idx=target_idx, target_side="enemies"),
        )

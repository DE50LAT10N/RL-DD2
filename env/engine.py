from __future__ import annotations

from dataclasses import dataclass, field
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
    stress_delta: int = 0
    relationships_delta: int = 0
    used_item: str | None = None


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
        hp = 36 if side == "heroes" else 24
        return Unit(
            id=f"{side[:1]}_{idx}_{archetype_id}",
            archetype_id=archetype_id,
            side="heroes" if side == "heroes" else "enemies",
            rank=idx + 1,
            hp=hp,
            max_hp=hp,
            stress=0,
        )

    def _effective_speed(self, unit: Unit) -> int:
        speed = 5
        for token in unit.tokens:
            if token.id == TokenId.SPEED:
                speed += token.count
            elif token.id == TokenId.DAZE:
                speed -= token.count
        return speed

    def _rebuild_speed_order(self, state: BattleState) -> None:
        alive = [u for u in state.heroes + state.enemies if u.alive and not u.is_remnant]
        scored = []
        for u in alive:
            jitter = self.rng.random() * 0.01
            scored.append((-(self._effective_speed(u) + jitter), u.id))
        scored.sort()
        state.speed_order = [uid for _, uid in scored]
        state.initiative = list(state.speed_order)

    def _advance_turn(self, state: BattleState) -> None:
        state.turn_idx += 1
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
            items_available={iid: 1 for iid in self.data.items.keys()},
            relationships={(a.id, b.id): 0 for a in heroes for b in heroes if a.id != b.id},
        )
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

    @staticmethod
    def _add_token(unit: Unit, token: TokenId, count: int = 1) -> None:
        for t in unit.tokens:
            if t.id == token:
                t.count += count
                return
        unit.tokens.append(Token(id=token, count=count))

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

    def _apply_start_turn_dot(self, unit: Unit) -> None:
        for token, dmg in ((TokenId.BURN, 2), (TokenId.BLEED, 2), (TokenId.POISON, 1)):
            if self._consume_once(unit, token):
                unit.hp -= dmg
        if unit.hp <= 0:
            self._handle_deaths_door(unit)

    def _handle_deaths_door(self, unit: Unit) -> None:
        if unit.is_remnant:
            unit.alive = False
            unit.hp = 0
            return
        if self._token_count(unit, TokenId.DEATHS_DOOR) > 0:
            if unit.side == "enemies":
                self._make_remnant(unit)
                return
            unit.alive = False
            unit.hp = 0
            return
        unit.hp = 1
        self._add_token(unit, TokenId.DEATHS_DOOR, 1)

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
        blind_penalty = 0.35 if self._token_count(actor, TokenId.BLIND) else 0.0
        dodge_bonus = 0.5 if self._token_count(target, TokenId.DODGE_PLUS) else (0.35 if self._token_count(target, TokenId.DODGE) else 0.0)
        hit = self.rng.random() > (blind_penalty + dodge_bonus)
        if self._token_count(actor, TokenId.BLIND):
            self._consume_once(actor, TokenId.BLIND)
        if self._token_count(target, TokenId.DODGE_PLUS):
            self._consume_once(target, TokenId.DODGE_PLUS)
        elif self._token_count(target, TokenId.DODGE):
            self._consume_once(target, TokenId.DODGE)
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
        base = self.rng.randint(skill.damage_lo, max(skill.damage_lo, skill.damage_hi))
        mult = 1.0
        if self._token_count(actor, TokenId.STRENGTH):
            mult *= 1.5
        if self._token_count(actor, TokenId.WEAK):
            mult *= 0.5
        if self._token_count(actor, TokenId.CRIT) or self.rng.random() < skill.crit_chance:
            mult *= 2.0
            self._consume_once(actor, TokenId.CRIT)
        if self._token_count(target, TokenId.VULNERABLE):
            mult *= 1.5
            self._consume_once(target, TokenId.VULNERABLE)
        dmg = max(0, int(round(base * mult)))
        if self._token_count(target, TokenId.BLOCK_PLUS):
            dmg = max(0, dmg - 4)
            self._consume_once(target, TokenId.BLOCK_PLUS)
        elif self._token_count(target, TokenId.BLOCK):
            dmg = max(0, dmg - 2)
            self._consume_once(target, TokenId.BLOCK)
        target.hp -= dmg
        if target.hp <= 0:
            self._handle_deaths_door(target)
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

    def _mark_terminal_if_done(self, state: BattleState) -> bool:
        if not any(self._combat_alive(u) for u in state.enemies):
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
            state.relationships[key] = state.relationships.get(key, 0) + event_delta

    def _end_round_decay(self, state: BattleState) -> None:
        for u in state.heroes + state.enemies:
            for tk in (TokenId.STRENGTH, TokenId.WEAK, TokenId.CRIT, TokenId.DAZE, TokenId.STUN):
                self._consume_once(u, tk)
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
        for rs in raw_skills[:4]:
            skills.append(Skill(
                id=rs.id,
                source_ranks=set(rs.source_ranks),
                target_ranks=set(rs.target_ranks),
                cooldown=rs.cooldown,
                is_friendly=rs.is_friendly,
                targets_self_party=rs.targets_self_party,
                move_self=rs.move_self,
                move_target=rs.move_target,
                damage_lo=rs.damage_lo,
                damage_hi=rs.damage_hi,
                crit_chance=rs.crit_chance,
                stress_damage=rs.stress_damage,
                heal=rs.heal,
                heal_stress=rs.heal_stress,
                costs=[Token(TokenId[k], v) for k, v in rs.costs.items() if k in TokenId.__members__],
                gives_self=[Token(TokenId[k], v) for k, v in rs.gives_self.items() if k in TokenId.__members__],
                gives_target=[Token(TokenId[k], v) for k, v in rs.gives_target.items() if k in TokenId.__members__],
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
        for si, sk in enumerate(skills[:4]):
            if state.skill_cooldowns.get((actor.id, sk.id), 0) > 0:
                continue
            if actor.rank not in sk.source_ranks:
                continue
            targets = state.heroes if sk.is_friendly else state.enemies
            target_side = "heroes" if sk.is_friendly else "enemies"
            candidate_targets = [
                (ti, tgt)
                for ti, tgt in enumerate(targets)
                if self._combat_alive(tgt) and tgt.rank in sk.target_ranks
            ]
            if not sk.is_friendly and not candidate_targets:
                candidate_targets = [
                    (ti, tgt)
                    for ti, tgt in enumerate(targets)
                    if tgt.alive and tgt.is_remnant and tgt.rank in sk.target_ranks
                ]
            for ti, tgt in candidate_targets:
                actions.append(ActionSpec(kind="skill", actor_idx=actor_idx, skill_idx=si, target_idx=ti, target_side=target_side))
        for iid, amount in state.items_available.items():
            if amount <= 0 or self._item_cooldowns.get(iid, 0) > 0:
                continue
            for ti, tgt in enumerate(state.heroes):
                if tgt.alive:
                    actions.append(ActionSpec(kind="item", actor_idx=actor_idx, item_id=iid, target_idx=ti, target_side="heroes"))
        if not self._token_count(actor, TokenId.IMMOBILIZE):
            for delta in (-1, 1):
                target_rank = actor.rank + delta
                if 1 <= target_rank <= len([h for h in state.heroes if h.alive]):
                    actions.append(ActionSpec(kind="move", actor_idx=actor_idx, move_delta=delta))
        actions.append(ActionSpec(kind="pass", actor_idx=actor_idx, target_idx=None))
        return actions

    def get_valid_actions(self, state: BattleState) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for a in self.legal_action_specs(state):
            if a.kind == "skill" and a.skill_idx is not None and a.target_idx is not None:
                pairs.append((a.skill_idx, a.target_idx))
        return pairs

    def _apply_item(self, state: BattleState, item: CombatItemSpec, actor: Unit, target: Unit) -> None:
        target.hp = min(target.max_hp, target.hp + item.heal)
        target.stress = max(0, target.stress - item.stress_heal)
        for tk, cnt in item.gives_target.items():
            if tk in TokenId.__members__:
                self._add_token(target, TokenId[tk], cnt)
        state.items_available[item.id] = max(0, state.items_available.get(item.id, 0) - 1)
        self._item_cooldowns[item.id] = item.cooldown
        self.last_trace = ActionTrace(actor_name=actor.id, skill_name="item", target_names=[target.id], healing_done=item.heal, used_item=item.id)

    def _pick_enemy_action(self, state: BattleState, actor: Unit) -> ActionSpec:
        skills = [s for s in self._skills_for(actor) if actor.rank in s.source_ranks and state.skill_cooldowns.get((actor.id, s.id), 0) <= 0]
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
        targetable_skills = [s for s in skills if target.rank in s.target_ranks]
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
            for tgt in state.enemies if actor.side == "heroes" else state.heroes:
                if self._combat_alive(tgt) and tgt.rank in sk.target_ranks:
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
            valid_targets = sum(1 for tgt in targets if self._combat_alive(tgt) and tgt.rank in sk.target_ranks)
            if valid_targets <= 0:
                continue
            # Favor positions with more reachable targets and stronger damage ceilings.
            score += valid_targets + (0.1 * max(0, sk.damage_hi))
        return score

    def _choose_reposition_delta(self, state: BattleState, actor: Unit) -> int:
        """Choose one-step rank shift that improves attack options the most."""
        current_rank = int(actor.rank)
        candidates: list[tuple[int, float]] = []
        for target_rank in (current_rank - 1, current_rank + 1):
            if not (1 <= target_rank <= 4):
                continue
            candidates.append((target_rank, self._attack_score_at_rank(state, actor, target_rank)))
        if not candidates:
            return 0
        best_rank, best_score = max(candidates, key=lambda x: (x[1], -abs(x[0] - 2.5)))
        current_score = self._attack_score_at_rank(state, actor, current_rank)
        if best_score <= current_score and self._has_attack_targets(state, actor):
            return 0
        return -1 if best_rank < current_rank else 1

    def execute_spec(self, state: BattleState, chosen: ActionSpec) -> BattleState:
        if state.done:
            return state
        side, actor_idx, actor = self.get_active_unit(state)
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
            if chosen not in legal:
                chosen = next((x for x in legal if x.kind == "skill"), next((x for x in legal if x.kind == "move"), ActionSpec(kind="pass", actor_idx=actor_idx)))
        else:
            chosen = self._pick_enemy_action(state, actor)

        if chosen.kind == "pass":
            self.last_trace.skill_name = "pass"
        elif chosen.kind == "move":
            self.last_trace.skill_name = "move"
            team = state.heroes if actor.side == "heroes" else state.enemies
            self._rank_shift(team, actor, int(chosen.move_delta))
        elif chosen.kind == "item" and chosen.item_id:
            target = state.heroes[min(chosen.target_idx or 0, len(state.heroes) - 1)]
            item = self.data.items[chosen.item_id]
            self._apply_item(state, item, actor, target)
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
                    target = self._resolve_guarded(state, target)
                    self.last_trace.skill_name = sk.id
                    self.last_trace.target_names = [target.id]
                    if self._roll_hit(actor, target):
                        if sk.damage_hi > 0:
                            dmg = self._apply_damage(actor, target, sk)
                            if target.side == "heroes":
                                self.last_trace.damage_to_heroes += dmg
                            else:
                                self.last_trace.damage_to_enemies += dmg
                        if sk.heal > 0:
                            target.hp = min(target.max_hp, target.hp + sk.heal)
                            self.last_trace.healing_done += sk.heal
                        if sk.heal_stress > 0:
                            target.stress = max(0, target.stress - sk.heal_stress)
                        target.stress = min(10, target.stress + sk.stress_damage)
                        for t in sk.gives_self:
                            self._add_token(actor, t.id, t.count)
                        for t in sk.gives_target:
                            self._add_token(target, t.id, t.count)
                        if self._token_count(target, TokenId.RIPOSTE) and target.side != actor.side:
                            self._consume_once(target, TokenId.RIPOSTE)
                            back = max(1, self.rng.randint(1, 4))
                            actor.hp -= back
                            if actor.side == "heroes":
                                self.last_trace.damage_to_heroes += back
                            else:
                                self.last_trace.damage_to_enemies += back
                            if actor.hp <= 0:
                                self._handle_deaths_door(actor)
                        if sk.cooldown > 0:
                            state.skill_cooldowns[(actor.id, sk.id)] = sk.cooldown
                        self._rank_shift(state.heroes if actor.side == "heroes" else state.enemies, actor, sk.move_self)
                        self._rank_shift(state.heroes if target.side == "heroes" else state.enemies, target, sk.move_target)
                        rel_delta = 1 if sk.heal > 0 else (-1 if sk.stress_damage > 0 else 0)
                        if rel_delta and actor.side == "heroes":
                            self._apply_relationship_event(state, actor, rel_delta)
                            self.last_trace.relationships_delta += rel_delta
                    for t in (TokenId.STRENGTH, TokenId.WEAK):
                        self._consume_once(actor, t)

        for u in state.heroes + state.enemies:
            if u.stress >= 10:
                u.afflicted = True

        self._mark_terminal_if_done(state)

        if state.speed_order:
            state.speed_order.pop(0)
        self._advance_turn(state)
        return state

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

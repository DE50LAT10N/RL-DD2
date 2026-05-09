# Game data loader and normalizer.
# Converts JSON fixtures and overrides into simulator-ready hero, monster, item, token, and encounter specs.
# Uses Pydantic for validation while allowing project-specific override files.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillSpec(BaseModel):
    id: str
    source_ranks: set[int] = Field(default_factory=lambda: {1, 2, 3, 4})
    target_ranks: set[int] = Field(default_factory=lambda: {1, 2, 3, 4})
    cooldown: int = 0
    charges: int = 0
    is_friendly: bool = False
    targets_self_party: bool = False
    target_self: bool = False
    multi_target: bool = False
    execution: int = 0
    damage_lo: int = 0
    damage_hi: int = 0
    crit_chance: float = 0.05
    stress_damage: int = 0
    heal: int = 0
    heal_percent: float = 0.0
    heal_threshold: float = 0.0
    heal_stress: int = 0
    cures_tokens: list[str] = Field(default_factory=list)
    move_self: int = 0
    move_target: int = 0
    dot_type: str = ""
    dot_amount: int = 0
    dot_duration: int = 3
    pierce: dict[str, float] = Field(default_factory=dict)
    costs: dict[str, int] = Field(default_factory=dict)
    gives_self: dict[str, int] = Field(default_factory=dict)
    gives_target: dict[str, int] = Field(default_factory=dict)
    combo_gives_self: dict[str, int] = Field(default_factory=dict)
    combo_gives_target: dict[str, int] = Field(default_factory=dict)
    combo_damage_multiplier: float = 1.0
    combo_dot_amount: int = 0
    combo_dot_duration: int = 0
    combo_consumes: bool = True
    extra_action_self: bool = False


class HeroArchetype(BaseModel):
    id: str
    name: str
    hp: int = 36
    speed: int = 5
    size: int = 1
    turns_per_round: int = 1
    move_distance: int = 1
    deathblow_resist: float = 0.6
    resistances: dict[str, float] = Field(default_factory=dict)
    skills: list[SkillSpec] = Field(default_factory=list)


class MonsterArchetype(BaseModel):
    id: str
    name: str
    hp: int = 24
    speed: int = 5
    size: int = 1
    turns_per_round: int = 1
    move_distance: int = 1
    death_armor: int = 0
    resistances: dict[str, float] = Field(default_factory=dict)
    skills: list[SkillSpec] = Field(default_factory=list)


class TokenSpec(BaseModel):
    id: str
    max_stacks: int = 3
    category: str = "neutral"
    is_dot: bool = False
    default_duration: int = 0
    default_max_stacks: int = 3


class CombatItemSpec(BaseModel):
    id: str
    name: str
    target_side: str = "heroes"
    cooldown: int = 1
    heal: int = 0
    stress_heal: int = 0
    gives_target: dict[str, int] = Field(default_factory=dict)


class EncounterSpec(BaseModel):
    id: str
    heroes: list[str] = Field(default_factory=list)
    enemies: list[str] = Field(default_factory=list)


class GameData(BaseModel):
    heroes: dict[str, HeroArchetype]
    monsters: dict[str, MonsterArchetype]
    tokens: dict[str, TokenSpec]
    items: dict[str, CombatItemSpec]
    encounters: dict[str, EncounterSpec]


DATA_ROOT = Path(os.getenv("APPDATA", "")) / "DDRL" / "data"
OVERRIDES_ROOT = Path(__file__).resolve().parent.parent / "configs" / "data_overrides"


def _load_json_files(folder: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    if not folder.exists():
        if strict:
            raise FileNotFoundError(f"Missing data folder: {folder}")
        return []
    out: list[dict[str, Any]] = []
    for file in sorted(folder.glob("*.json")):
        try:
            out.append(json.loads(file.read_text(encoding="utf-8")))
        except Exception as exc:
            if strict:
                raise ValueError(f"Invalid JSON in {file}: {exc}") from exc
    return out


def _coerce_skill(raw: dict[str, Any], idx: int) -> SkillSpec:
    sid = str(raw.get("id") or raw.get("Id") or raw.get("name") or f"skill_{idx}")
    return SkillSpec(
        id=sid,
        source_ranks=set(raw.get("source_ranks") or raw.get("SourceRanks") or [1, 2, 3, 4]),
        target_ranks=set(raw.get("target_ranks") or raw.get("TargetRanks") or [1, 2, 3, 4]),
        cooldown=int(raw.get("cooldown") or raw.get("Cooldown") or 0),
        charges=int(raw.get("charges") or raw.get("Charges") or 0),
        is_friendly=bool(raw.get("is_friendly") or raw.get("IsFriendly") or False),
        targets_self_party=bool(raw.get("targets_self_party") or raw.get("TargetsSelfParty") or raw.get("target_self_party") or False),
        target_self=bool(raw.get("target_self") or raw.get("TargetSelf") or False),
        multi_target=bool(raw.get("multi_target") or raw.get("MultiTarget") or False),
        execution=int(raw.get("execution") or raw.get("Execution") or 0),
        damage_lo=int(raw.get("damage_lo") or raw.get("DamageLo") or 0),
        damage_hi=int(raw.get("damage_hi") or raw.get("DamageHi") or raw.get("damage_lo") or 0),
        crit_chance=float(raw.get("crit_chance") or raw.get("CritChance") or 0.05),
        stress_damage=int(raw.get("stress_damage") or raw.get("StressDamage") or 0),
        heal=int(raw.get("heal") or raw.get("Heal") or 0),
        heal_percent=float(raw.get("heal_percent") or raw.get("HealPercent") or 0.0),
        heal_threshold=float(raw.get("heal_threshold") or raw.get("HealThreshold") or 0.0),
        heal_stress=int(raw.get("heal_stress") or raw.get("HealStress") or 0),
        cures_tokens=[str(t).upper() for t in (raw.get("cures_tokens") or raw.get("CuresTokens") or [])],
        move_self=int(raw.get("move_self") or raw.get("MoveSelf") or 0),
        move_target=int(raw.get("move_target") or raw.get("MoveTarget") or 0),
        dot_type=str(raw.get("dot_type") or raw.get("DotType") or "").upper(),
        dot_amount=int(raw.get("dot_amount") or raw.get("DotAmount") or 0),
        dot_duration=int(raw.get("dot_duration") or raw.get("DotDuration") or 3),
        pierce={str(k).lower(): float(v) for k, v in (raw.get("pierce") or raw.get("Pierce") or {}).items()},
        costs={str(k): int(v) for k, v in (raw.get("costs") or raw.get("Costs") or {}).items()},
        gives_self={str(k): int(v) for k, v in (raw.get("gives_self") or raw.get("GivesSelf") or {}).items()},
        gives_target={str(k): int(v) for k, v in (raw.get("gives_target") or raw.get("GivesTarget") or {}).items()},
        combo_gives_self={str(k): int(v) for k, v in (raw.get("combo_gives_self") or raw.get("ComboGivesSelf") or {}).items()},
        combo_gives_target={str(k): int(v) for k, v in (raw.get("combo_gives_target") or raw.get("ComboGivesTarget") or {}).items()},
        combo_damage_multiplier=float(raw.get("combo_damage_multiplier") or raw.get("ComboDamageMultiplier") or 1.0),
        combo_dot_amount=int(raw.get("combo_dot_amount") or raw.get("ComboDotAmount") or 0),
        combo_dot_duration=int(raw.get("combo_dot_duration") or raw.get("ComboDotDuration") or 0),
        combo_consumes=bool(raw.get("combo_consumes", raw.get("ComboConsumes", True))),
        extra_action_self=bool(raw.get("extra_action_self") or raw.get("ExtraActionSelf") or False),
    )


def _defaults() -> GameData:
    strike = SkillSpec(id="strike", damage_lo=4, damage_hi=7)
    heal = SkillSpec(id="bandage", heal=5, target_ranks={1, 2, 3, 4})
    heroes = {
        "man_at_arms": HeroArchetype(id="man_at_arms", name="Man-at-Arms", hp=36, speed=3, move_distance=1, skills=[strike]),
        "plague_doctor": HeroArchetype(id="plague_doctor", name="Plague Doctor", hp=29, speed=4, move_distance=1, skills=[strike, heal]),
    }
    monsters = {
        "ghoul": MonsterArchetype(id="ghoul", name="Ghoul", hp=36, speed=4, death_armor=1, move_distance=2, skills=[SkillSpec(id="claw", damage_lo=3, damage_hi=6)]),
        "cultist": MonsterArchetype(id="cultist", name="Cultist", hp=24, speed=3, move_distance=1, skills=[SkillSpec(id="stab", damage_lo=2, damage_hi=5)]),
    }
    tokens = {k: TokenSpec(id=k) for k in [
        "BLOCK", "BLOCK_PLUS", "DODGE", "DODGE_PLUS", "CRIT", "STRENGTH", "WEAK", "WINDED", "DAZE", "STUN", "COMBO",
        "TAUNT", "GUARDED", "STEALTH", "RIPOSTE", "SPEED", "VULNERABLE", "BLIND", "BURN", "BLEED", "BLIGHT",
        "POISON", "REGENERATION", "HORROR", "DEATHS_DOOR", "DEATH_ARMOR", "UNSTOPPABLE", "IMMOBILIZE",
    ]}
    items = {
        "smoke_bomb": CombatItemSpec(id="smoke_bomb", name="Smoke Bomb", target_side="enemies", gives_target={"BLIND": 1}),
        "healing_salve": CombatItemSpec(id="healing_salve", name="Healing Salve", heal=5),
    }
    encounters = {
        "road_fight": EncounterSpec(id="road_fight", heroes=["man_at_arms", "plague_doctor"], enemies=["ghoul", "cultist"])
    }
    return GameData(heroes=heroes, monsters=monsters, tokens=tokens, items=items, encounters=encounters)


def _merge_overrides(data: GameData, *, strict: bool = False) -> GameData:
    if not OVERRIDES_ROOT.exists():
        return data
    for file in sorted(OVERRIDES_ROOT.glob("*.json")):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:
            if strict:
                raise ValueError(f"Invalid override JSON in {file}: {exc}") from exc
            continue
        for hid, raw in payload.get("heroes", {}).items():
            merged = data.heroes.get(hid, HeroArchetype(id=hid, name=hid)).model_dump()
            merged.update(raw)
            data.heroes[hid] = HeroArchetype.model_validate(merged)
        for mid, raw in payload.get("monsters", {}).items():
            merged = data.monsters.get(mid, MonsterArchetype(id=mid, name=mid)).model_dump()
            merged.update(raw)
            data.monsters[mid] = MonsterArchetype.model_validate(merged)
        for tid, raw in payload.get("tokens", {}).items():
            merged = data.tokens.get(tid, TokenSpec(id=tid)).model_dump()
            merged.update(raw)
            data.tokens[tid] = TokenSpec.model_validate(merged)
        for iid, raw in payload.get("items", {}).items():
            merged = data.items.get(iid, CombatItemSpec(id=iid, name=iid)).model_dump()
            merged.update(raw)
            data.items[iid] = CombatItemSpec.model_validate(merged)
        for eid, raw in payload.get("encounters", {}).items():
            merged = data.encounters.get(eid, EncounterSpec(id=eid)).model_dump()
            merged.update(raw)
            data.encounters[eid] = EncounterSpec.model_validate(merged)
    return data


def load(data_root: Path | None = None, *, strict: bool = False) -> GameData:
    root = data_root or DATA_ROOT
    if not root.exists():
        if strict:
            raise FileNotFoundError(f"Data root does not exist: {root}")
        return _merge_overrides(_defaults(), strict=strict)

    heroes_raw = _load_json_files(root / "heroes", strict=strict)
    monsters_raw = _load_json_files(root / "monsters", strict=strict)
    tokens_raw = _load_json_files(root / "tokens", strict=strict)
    items_raw = _load_json_files(root / "items", strict=strict)
    encounters_raw = _load_json_files(root / "encounters", strict=strict)

    data = _defaults()

    for i, h in enumerate(heroes_raw):
        hid = str(h.get("id") or h.get("Id") or h.get("name") or h.get("Name") or f"hero_{i}")
        skills = [_coerce_skill(s, si) for si, s in enumerate(h.get("skills") or h.get("Skills") or [])]
        data.heroes[hid] = HeroArchetype(
            id=hid,
            name=str(h.get("name") or h.get("Name") or hid),
            hp=int(h.get("hp") or h.get("HP") or h.get("max_hp") or h.get("MaxHp") or 36),
            speed=int(h.get("speed") or h.get("Speed") or 5),
            size=int(h.get("size") or h.get("Size") or 1),
            turns_per_round=int(h.get("turns_per_round") or h.get("TurnsPerRound") or h.get("turns") or h.get("Turns") or 1),
            move_distance=int(h.get("move_distance") or h.get("MoveDistance") or 1),
            deathblow_resist=float(h.get("deathblow_resist") or h.get("DeathblowResist") or 0.6),
            resistances={str(k).lower(): float(v) for k, v in (h.get("resistances") or h.get("Resistances") or {}).items()},
            skills=skills,
        )

    for i, m in enumerate(monsters_raw):
        mid = str(m.get("id") or m.get("Id") or m.get("name") or m.get("Name") or f"monster_{i}")
        skills = [_coerce_skill(s, si) for si, s in enumerate(m.get("skills") or m.get("Skills") or [])]
        data.monsters[mid] = MonsterArchetype(
            id=mid,
            name=str(m.get("name") or m.get("Name") or mid),
            hp=int(m.get("hp") or m.get("HP") or m.get("max_hp") or m.get("MaxHp") or 24),
            speed=int(m.get("speed") or m.get("Speed") or 5),
            size=int(m.get("size") or m.get("Size") or 1),
            turns_per_round=int(m.get("turns_per_round") or m.get("TurnsPerRound") or m.get("turns") or m.get("Turns") or 1),
            move_distance=int(m.get("move_distance") or m.get("MoveDistance") or 1),
            death_armor=int(m.get("death_armor") or m.get("DeathArmor") or 0),
            resistances={str(k).lower(): float(v) for k, v in (m.get("resistances") or m.get("Resistances") or {}).items()},
            skills=skills,
        )

    for i, t in enumerate(tokens_raw):
        tid = str(t.get("id") or t.get("Id") or t.get("name") or t.get("Name") or f"token_{i}")
        max_stacks = int(t.get("max_stacks") or t.get("MaxStacks") or 3)
        data.tokens[tid] = TokenSpec(
            id=tid,
            max_stacks=max_stacks,
            category=str(t.get("category") or t.get("Category") or "neutral"),
            is_dot=bool(t.get("is_dot") or t.get("IsDot") or False),
            default_duration=int(t.get("default_duration") or t.get("DefaultDuration") or 0),
            default_max_stacks=int(t.get("default_max_stacks") or t.get("DefaultMaxStacks") or max_stacks),
        )

    for i, it in enumerate(items_raw):
        iid = str(it.get("id") or it.get("Id") or it.get("name") or it.get("Name") or f"item_{i}")
        data.items[iid] = CombatItemSpec(
            id=iid,
            name=str(it.get("name") or it.get("Name") or iid),
            target_side=str(it.get("target_side") or it.get("TargetSide") or ("enemies" if it.get("gives_target") or it.get("GivesTarget") else "heroes")),
            cooldown=int(it.get("cooldown") or it.get("Cooldown") or 1),
            heal=int(it.get("heal") or it.get("Heal") or 0),
            stress_heal=int(it.get("stress_heal") or it.get("StressHeal") or 0),
            gives_target={str(k): int(v) for k, v in (it.get("gives_target") or it.get("GivesTarget") or {}).items()},
        )

    for i, e in enumerate(encounters_raw):
        eid = str(e.get("id") or e.get("Id") or e.get("name") or e.get("Name") or f"encounter_{i}")
        data.encounters[eid] = EncounterSpec(
            id=eid,
            heroes=[str(x) for x in (e.get("heroes") or e.get("Heroes") or [])],
            enemies=[str(x) for x in (e.get("enemies") or e.get("Enemies") or [])],
        )

    return _merge_overrides(data, strict=strict)

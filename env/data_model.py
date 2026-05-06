from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TokenId(str, Enum):
    BLOCK = "BLOCK"
    BLOCK_PLUS = "BLOCK_PLUS"
    DODGE = "DODGE"
    DODGE_PLUS = "DODGE_PLUS"
    CRIT = "CRIT"
    STRENGTH = "STRENGTH"
    WEAK = "WEAK"
    DAZE = "DAZE"
    STUN = "STUN"
    COMBO = "COMBO"
    TAUNT = "TAUNT"
    RIPOSTE = "RIPOSTE"
    SPEED = "SPEED"
    VULNERABLE = "VULNERABLE"
    BLIND = "BLIND"
    BURN = "BURN"
    BLEED = "BLEED"
    POISON = "POISON"
    DEATHS_DOOR = "DEATHS_DOOR"
    UNSTOPPABLE = "UNSTOPPABLE"
    IMMOBILIZE = "IMMOBILIZE"


@dataclass(slots=True)
class Token:
    id: TokenId
    count: int = 1


@dataclass(slots=True)
class Unit:
    id: str
    archetype_id: str
    side: Literal["heroes", "enemies"]
    rank: int
    hp: int
    max_hp: int
    stress: int = 0
    tokens: list[Token] = field(default_factory=list)
    alive: bool = True
    afflicted: bool = False
    is_remnant: bool = False


@dataclass(slots=True)
class Skill:
    id: str
    source_ranks: set[int]
    target_ranks: set[int]
    cooldown: int = 0
    is_friendly: bool = False
    targets_self_party: bool = False
    move_self: int = 0
    move_target: int = 0
    damage_lo: int = 0
    damage_hi: int = 0
    crit_chance: float = 0.05
    costs: list[Token] = field(default_factory=list)
    gives_self: list[Token] = field(default_factory=list)
    gives_target: list[Token] = field(default_factory=list)
    stress_damage: int = 0
    heal: int = 0
    heal_stress: int = 0


@dataclass(slots=True)
class ActionSpec:
    kind: Literal["skill", "item", "move", "pass"]
    actor_idx: int = 0
    skill_idx: int | None = None
    item_id: str | None = None
    target_idx: int | None = None
    target_side: Literal["heroes", "enemies"] | None = None
    move_delta: int = 0


@dataclass(slots=True)
class BattleState:
    heroes: list[Unit]
    enemies: list[Unit]
    round: int
    initiative: list[str]
    turn_idx: int
    speed_order: list[str] = field(default_factory=list)
    skill_cooldowns: dict[tuple[str, str], int] = field(default_factory=dict)
    items_available: dict[str, int] = field(default_factory=dict)
    relationships: dict[tuple[str, str], int] = field(default_factory=dict)
    done: bool = False
    heroes_won: bool | None = None

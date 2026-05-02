from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.data_model import Token, TokenId
from env.engine import SimBattleBackend


def test_block_flat_reduction() -> None:
    b = SimBattleBackend(seed=7)
    s = b.reset(seed=7)
    s.enemies[0].tokens = [Token(TokenId.BLOCK, 1)]
    pre = s.enemies[0].hp
    b.execute_action(s, 0, 0)
    assert pre - s.enemies[0].hp == 8
    assert not any(t.id == TokenId.BLOCK for t in s.enemies[0].tokens)


def test_dodge_chain_attack() -> None:
    b = SimBattleBackend(seed=11)
    s = b.reset(seed=11)
    s.enemies[0].tokens = [Token(TokenId.DODGE, 1)]
    b.execute_action(s, 0, 0)
    had_dodge = any(t.id == TokenId.DODGE for t in s.enemies[0].tokens)
    assert not had_dodge


def test_riposte_loop() -> None:
    b = SimBattleBackend(seed=13)
    s = b.reset(seed=13)
    s.enemies[0].tokens = [Token(TokenId.RIPOSTE, 1)]
    pre = s.heroes[0].hp
    b.execute_action(s, 0, 0)
    assert s.heroes[0].hp <= pre


def test_deaths_door_kill() -> None:
    b = SimBattleBackend(seed=17)
    s = b.reset(seed=17)
    s.enemies[0].hp = 1
    b.execute_action(s, 0, 0)
    assert any(t.id == TokenId.DEATHS_DOOR for t in s.enemies[0].tokens) or not s.enemies[0].alive
    s.turn_idx = 0
    b.execute_action(s, 0, 0)
    assert not s.enemies[0].alive


def test_stress_meltdown() -> None:
    b = SimBattleBackend(seed=19)
    s = b.reset(seed=19)
    s.heroes[0].stress = 10
    b.execute_action(s, 0, 0)
    assert s.heroes[0].afflicted


def main() -> int:
    test_block_flat_reduction()
    test_dodge_chain_attack()
    test_riposte_loop()
    test_deaths_door_kill()
    test_stress_meltdown()
    print("deterministic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

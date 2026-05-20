# Live PPO runner for Darkest Dungeon II.
# Connects to the BepInEx plugin, builds observations from live state, and sends chosen actions.
# Keeps strategy in the model while using DD2 legal_actions as the validity boundary.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MOD_AGENT_DIR = PROJECT_ROOT / "mod" / "agent"
if str(MOD_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(MOD_AGENT_DIR))

from env.data_model import ActionSpec
from env.dd_env import (
    ACTION_SPACE_SIZE,
    ACTION_TARGET_SLOTS,
    MAX_SKILLS_PER_UNIT,
    DarkestDungeonEnv,
)
from env.live.mod_state import parse_state
from dd2_env import DD2Env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained RL agent in live DD2.")
    p.add_argument("--model", default="runs/best/best_model.zip")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--reset-timeout", type=float, default=120.0)
    p.add_argument("--action-timeout", type=float, default=10.0)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--step-delay", type=float, default=0.25, help="Delay between live actions to avoid flooding game state.")
    p.add_argument("--mode", choices=("pass_only", "hook_only", "ppo", "qrdqn"), default="ppo", help="Force action source for commit-path diagnostics.")
    p.add_argument("--max-no-ack-retries", type=int, default=3, help="Max empty recv cycles before step returns no_ack.")
    p.add_argument("--enemy-turn-wait", type=float, default=60.0, help="Seconds to wait for the next hero turn before sending an action.")
    p.add_argument("--wait-poll", type=float, default=0.5, help="Polling interval while waiting for enemy/animation phases to finish.")
    p.add_argument("--legal-action-wait", type=float, default=8.0, help="Seconds to wait for a hero action frame with non-empty legal_actions.")
    p.add_argument("--legal-stable-polls", type=int, default=3, help="Consecutive equal legal action frames required before acting.")
    p.add_argument("--post-action-settle", type=float, default=0.75, help="Minimum delay after a committed live action before polling again.")
    p.add_argument("--stunned-turn-wait", type=float, default=12.0, help="Seconds to wait for DD2 to auto-skip a stunned active unit.")
    p.add_argument("--terminal-wait", type=float, default=10.0, help="Seconds to wait for battle_end before safe-stop exits.")
    p.add_argument("--terminal-settle", type=float, default=3.0, help="Seconds to keep the live connection open after battle end.")
    p.add_argument("--act-on-enemy-turn", action="store_true", help="Diagnostic override: send actions even when active_side is not heroes.")
    p.add_argument("--log-legal-actions", action="store_true", help="Print live legal actions and encoded mask indices before each model choice.")
    p.add_argument(
        "--max-stuck-live-actions",
        type=int,
        default=8,
        help="Exit after this many consecutive live steps with no battle progress (no-delta / move_unavailable / pass unavailable).",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--action-log", default="", help="Optional JSONL path for live state/action/outcome records.")
    return p.parse_args()


def _settle_after_terminal(args: argparse.Namespace) -> None:
    if args.terminal_settle > 0:
        import time
        time.sleep(args.terminal_settle)


def _maybe_print_terminal(live_env: DD2Env, *, steps: int, total_reward: float, args: argparse.Namespace, stop_reason: str) -> tuple[bool, dict[str, Any]]:
    obs, done, info = live_env.wait_for_terminal(max_wait=args.terminal_wait, poll_interval=args.wait_poll)
    if done:
        print(
            f"episode_end steps={steps} total_reward={total_reward:.3f} "
            f"heroes_won={info.get('heroes_won')} method={info.get('method')} "
            f"terminal_source={info.get('terminal_source')}",
            flush=True,
        )
        _settle_after_terminal(args)
        return True, info
    print(
        f"{stop_reason}_no_terminal "
        f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
        f"heroes_alive={info.get('heroes_alive')} enemies_alive={info.get('enemies_alive')}",
        flush=True,
    )
    return False, info


def _live_state_from_obs(obs: dict[str, Any]) -> dict[str, Any]:
    # parse_state() accepts top-level state fields.
    items_available = {
        str(k): int(v)
        for k, v in (obs.get("items_available") or {}).items()
        if str(k)
    }
    for raw in obs.get("legal_actions") or []:
        item_id = str(raw.get("item_id") or "")
        if item_id:
            items_available.setdefault(item_id, 1)
    return {
        "heroes": obs.get("heroes", []),
        "enemies": obs.get("enemies", []),
        "round": obs.get("round", 0),
        "active_index": obs.get("active_index", 0),
        "active_side": obs.get("active_side", "none"),
        "items_available": items_available,
        "done": obs.get("done", False),
        "heroes_won": obs.get("heroes_won"),
    }


def _token_id_text(token: dict[str, Any]) -> str:
    return " ".join(
        str(token.get(key, ""))
        for key in ("id", "name", "token_id", "TokenId", "actor_data_id")
    ).lower()


def _unit_has_stun_token(unit: dict[str, Any]) -> bool:
    for token in unit.get("tokens") or []:
        text = _token_id_text(token)
        if "stun" in text or "РѕРіР»СѓС€" in text:
            try:
                count = int(token.get("count", token.get("stacks", 1)))
            except Exception:
                count = 1
            if count > 0:
                return True
    return False


def _active_unit(live_obs: dict[str, Any]) -> dict[str, Any] | None:
    side = str(live_obs.get("active_side") or "none")
    active_index = int(live_obs.get("active_index", -1))
    units = live_obs.get("heroes") if side == "heroes" else live_obs.get("enemies")
    for idx, unit in enumerate(units or []):
        slot = int(unit.get("slot", idx))
        if slot == active_index:
            return unit
    return None


def _active_unit_is_stunned(live_obs: dict[str, Any]) -> bool:
    unit = _active_unit(live_obs)
    return bool(unit and _unit_has_stun_token(unit))


def _build_mask_and_legal(env: DarkestDungeonEnv, live_obs: dict[str, Any]) -> tuple[np.ndarray, list[ActionSpec]]:
    state = parse_state(_live_state_from_obs(live_obs))
    env.state = state
    legal: list[ActionSpec] = []
    raw_actions = live_obs.get("legal_actions") or []
    for raw in raw_actions:
        if raw.get("pass_turn"):
            legal.append(ActionSpec(kind="pass"))
            continue
        if "move_delta" in raw:
            legal.append(ActionSpec(kind="move", move_delta=int(raw.get("move_delta", 0))))
            continue
        if "item_id" in raw:
            target_team = str(raw.get("target_team") or "heroes")
            target_idx = int(raw.get("target_idx", 0))
            if not (0 <= target_idx < ACTION_TARGET_SLOTS):
                continue
            legal.append(
                ActionSpec(
                    kind="item",
                    item_id=str(raw.get("item_id") or ""),
                    target_idx=target_idx,
                    target_side=target_team,
                )
            )
            continue
        if "skill_idx" in raw and "target_idx" in raw:
            skill_idx = int(raw.get("skill_idx", 0))
            target_team = str(raw.get("target_team") or "enemies")
            target_idx = int(raw.get("target_idx", 0))
            if not (0 <= skill_idx < MAX_SKILLS_PER_UNIT and 0 <= target_idx < ACTION_TARGET_SLOTS):
                continue
            legal.append(
                ActionSpec(
                    kind="skill",
                    skill_idx=skill_idx,
                    target_idx=target_idx,
                    target_side=target_team,
                )
            )

    mask = np.zeros((ACTION_SPACE_SIZE,), dtype=bool)
    for spec in legal:
        idx = env._encode_action(spec)
        if 0 <= idx < ACTION_SPACE_SIZE:
            mask[idx] = True
    return mask, legal


def _spec_to_payload(spec: ActionSpec, live_obs: dict[str, Any]) -> dict[str, Any]:
    if spec.kind == "pass":
        return {"pass_turn": True}
    hero_slot = int(live_obs.get("active_index", 0))
    if spec.kind == "move":
        delta = int(spec.move_delta)
        payload: dict[str, Any] = {"hero_slot": hero_slot, "move_delta": delta}
        for raw in live_obs.get("legal_actions") or []:
            if "move_delta" not in raw or int(raw.get("move_delta", 0)) != delta:
                continue
            for forwarded_key in ("move_skill_id", "target_idx", "target_team"):
                if forwarded_key in raw:
                    payload[forwarded_key] = raw[forwarded_key]
            return payload
        return payload
    if spec.kind == "skill":
        return {
            "hero_slot": hero_slot,
            "skill_idx": int(spec.skill_idx or 0),
            "target_idx": int(spec.target_idx or 0),
            "target_team": str(spec.target_side or "enemies"),
        }
    if spec.kind == "item":
        return {
            "hero_slot": hero_slot,
            "item_id": str(spec.item_id or ""),
            "target_idx": int(spec.target_idx or 0),
            "target_team": str(spec.target_side or "heroes"),
        }
    raise ValueError(f"Unsupported action kind: {spec.kind}")


def _build_model_mask(
    env: DarkestDungeonEnv,
    legal: list[ActionSpec],
    model_action_dim: int,
    live_obs: dict[str, Any],
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    if model_action_dim != ACTION_SPACE_SIZE:
        raise ValueError(f"Live model requires current action space: model_dim={model_action_dim} env_dim={ACTION_SPACE_SIZE}")
    mask = np.zeros((model_action_dim,), dtype=bool)
    payload_by_idx: dict[int, dict[str, Any]] = {}
    for spec in legal:
        idx = env._encode_action(spec)
        if 0 <= idx < model_action_dim:
            mask[idx] = True
            payload = _spec_to_payload(spec, live_obs)
            payload_by_idx.setdefault(idx, payload)
    return mask, payload_by_idx


_LIVE_STUCK_REASONS = frozenset(
    {
        "move_unavailable",
        "move_skill_id_missing",
        "move_target_actor_missing",
        "move_target_rank_invalid",
        "selected_move_path_no_delta",
        "pass_turn_unavailable",
        "no_post_state",
        "no_state_delta_after_ack",
        "selected_skill_path_no_delta",
        "no_state_delta_after_commit",
        "no_ack",
    }
)


def _turn_key(live_obs: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(live_obs.get("active_side", "none")),
        int(live_obs.get("active_index", -1)),
        int(live_obs.get("round", 0)),
    )


def _compact_legal(env: DarkestDungeonEnv, legal: list[ActionSpec]) -> list[dict[str, Any]]:
    return [
        {
            "idx": env._encode_action(spec),
            "kind": spec.kind,
            "skill_idx": spec.skill_idx,
            "item_id": spec.item_id,
            "target_idx": spec.target_idx,
            "target_side": spec.target_side,
            "move_delta": spec.move_delta,
        }
        for spec in legal
    ]


def _legal_fingerprint(env: DarkestDungeonEnv, legal: list[ActionSpec]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                env._encode_action(spec),
                spec.kind,
                spec.skill_idx,
                spec.item_id,
                spec.target_idx,
                spec.target_side,
                spec.move_delta,
            )
            for spec in legal
        )
    )


def _compact_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for idx, unit in enumerate(units or []):
        compact.append(
            {
                "slot": int(unit.get("slot", idx) or idx),
                "rank": int(unit.get("rank", unit.get("slot", idx + 1)) or (idx + 1)),
                "id": str(unit.get("id") or unit.get("actor_id") or ""),
                "name": str(unit.get("name") or unit.get("display_name") or ""),
                "archetype_id": str(unit.get("archetype_id") or unit.get("hero_id") or unit.get("monster_id") or ""),
                "hp": int(unit.get("hp", 0) or 0),
                "max_hp": int(unit.get("max_hp", unit.get("hp", 1)) or 1),
                "stress": int(unit.get("stress", 0) or 0),
                "alive": bool(unit.get("alive", True)),
                "tokens": unit.get("tokens") or [],
            }
        )
    return compact


def _compact_live_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": obs.get("round"),
        "active_side": obs.get("active_side"),
        "active_index": obs.get("active_index"),
        "heroes": _compact_units(obs.get("heroes") or []),
        "enemies": _compact_units(obs.get("enemies") or []),
        "done": bool(obs.get("done", False)),
        "heroes_won": obs.get("heroes_won"),
    }


def _append_action_log(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _wait_for_stunned_turn_skip(
    live_env: DD2Env,
    obs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool]:
    """Wait for DD2 to consume a stunned unit turn without sending an action."""
    import time

    initial_key = _turn_key(obs)
    deadline = time.time() + max(0.0, args.stunned_turn_wait)
    while time.time() < deadline and not obs.get("done"):
        print(
            "waiting_for_stunned_turn_skip "
            f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
            flush=True,
        )
        time.sleep(max(0.05, args.wait_poll))
        obs = live_env.refresh(max_wait=max(0.25, args.wait_poll), timeout=0.25)
        if obs.get("done") or _turn_key(obs) != initial_key:
            return obs, True
    return obs, False


def _wait_for_hero_legal_actions(
    live_env: DD2Env,
    shadow_env: DarkestDungeonEnv,
    obs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[ActionSpec]]:
    """Wait for DD2 to expose a stable hero action frame.

    Right after animations/enemy turns, active_side can already be "heroes"
    while GetValidSkillTargetEntries is empty or changing. Stress breaks and
    relationship actouts can also briefly expose legal actions before the UI is
    settled, so require the same turn and same legal list for a few polls.
    """
    import time

    deadline = time.time() + max(0.0, args.legal_action_wait)
    last_report = 0.0
    required_stable = max(1, int(args.legal_stable_polls))
    stable_count = 0
    last_key: tuple[str, int, int] | None = None
    last_fingerprint: tuple[tuple[Any, ...], ...] | None = None
    last_legal: list[ActionSpec] = []
    while not obs.get("done") and obs.get("active_side") == "heroes":
        _, legal = _build_mask_and_legal(shadow_env, obs)
        last_legal = legal
        if legal:
            key = _turn_key(obs)
            fingerprint = _legal_fingerprint(shadow_env, legal)
            if key == last_key and fingerprint == last_fingerprint:
                stable_count += 1
            else:
                stable_count = 1
                last_key = key
                last_fingerprint = fingerprint
            if stable_count >= required_stable:
                return obs, legal
        else:
            stable_count = 0
            last_key = None
            last_fingerprint = None
        if time.time() >= deadline:
            if legal:
                print(
                    "legal_action_wait_timeout_using_latest "
                    f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                    f"stable={stable_count}/{required_stable}",
                    flush=True,
                )
            return obs, legal
        now = time.time()
        if now - last_report >= 1.0:
            if legal:
                print(
                    "waiting_for_stable_legal_actions "
                    f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                    f"stable={stable_count}/{required_stable}",
                    flush=True,
                )
            else:
                print(
                    "waiting_for_legal_actions "
                    f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                    flush=True,
                )
            last_report = now
        time.sleep(max(0.05, args.wait_poll))
        obs = live_env.refresh(max_wait=max(0.25, args.wait_poll), timeout=0.25)

    _, legal = _build_mask_and_legal(shadow_env, obs)
    return obs, legal or last_legal


def _decode_to_payload(
    env: DarkestDungeonEnv,
    action_idx: int,
    live_obs: dict[str, Any],
    model_action_dim: int,
    payload_by_idx: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if payload_by_idx and int(action_idx) in payload_by_idx:
        return dict(payload_by_idx[int(action_idx)])
    if model_action_dim != ACTION_SPACE_SIZE:
        raise ValueError(f"Live model requires current action space: model_dim={model_action_dim} env_dim={ACTION_SPACE_SIZE}")
    spec = env._decode_action(int(action_idx))
    if spec.kind == "pass":
        return {"pass_turn": True}

    hero_slot = int(live_obs.get("active_index", 0))
    if spec.kind == "move":
        return _spec_to_payload(spec, live_obs)
    if spec.kind == "skill":
        return {
            "hero_slot": hero_slot,
            "skill_idx": int(spec.skill_idx or 0),
            "target_idx": int(spec.target_idx or 0),
            "target_team": str(spec.target_side or "enemies"),
        }
    if spec.kind == "item":
        return {
            "hero_slot": hero_slot,
            "item_id": str(spec.item_id or ""),
            "target_idx": int(spec.target_idx or 0),
            "target_team": str(spec.target_side or "heroes"),
        }
    raise ValueError(f"Unsupported action kind: {spec.kind}")


def _adapt_obs_for_model(obs_vec: np.ndarray, expected_dim: int) -> np.ndarray:
    current_dim = int(obs_vec.shape[0])
    if current_dim == expected_dim:
        return obs_vec
    raise ValueError(f"Live model requires current observation space: model_dim={expected_dim} env_dim={current_dim}")


def _set_shadow_action_history(env: DarkestDungeonEnv, consecutive_moves: int, last_skill_name: str) -> None:
    # This mirrors training-only reward history into the live observation. It is
    # not a mask or override: the policy still receives all legal actions.
    env.reward_calculator._consecutive_moves = int(max(0, consecutive_moves))
    env.reward_calculator._last_skill_name = str(last_skill_name or "")


def main() -> int:
    args = parse_args()

    shadow_env: DarkestDungeonEnv | None = None
    agent = None
    model_obs_dim = 0
    model_action_dim = 0
    model_modes = {"ppo", "qrdqn"}
    if args.mode in model_modes:
        if args.mode == "ppo":
            from agents.ppo_agent import PPOAgent as AgentClass
        else:
            from agents.qrdqn_agent import QRDQNAgent as AgentClass

        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        # Shadow env is needed for model observation/action encoding.
        shadow_env = DarkestDungeonEnv(seed=7)
        agent = AgentClass.load(model_path)
        model_obs_dim = int(agent.model.observation_space.shape[0])
        model_action_dim = int(agent.model.action_space.n)
        if model_obs_dim != int(shadow_env.observation_space.shape[0]):
            raise ValueError(
                f"Live {args.mode} requires a model trained with the current observation space. "
                f"model_dim={model_obs_dim} env_dim={shadow_env.observation_space.shape[0]}"
            )
        if model_action_dim != int(shadow_env.action_space.n):
            raise ValueError(
                f"Live {args.mode} requires a model trained with the current action space. "
                f"model_dim={model_action_dim} env_dim={shadow_env.action_space.n}"
            )

    action_log_path = Path(args.action_log) if args.action_log else None
    if action_log_path is not None and not action_log_path.is_absolute():
        action_log_path = PROJECT_ROOT / action_log_path

    live_env = DD2Env(
        host=args.host,
        port=args.port,
        reset_timeout=args.reset_timeout,
        action_timeout=args.action_timeout,
        max_no_ack_retries=args.max_no_ack_retries,
        verbose=not args.quiet,
    )
    try:
        obs = live_env.reset()
        total_reward = 0.0
        stuck_live_actions = 0
        prev_stuck_turn: tuple[str, int, int] | None = None
        live_consecutive_moves = 0
        live_last_skill_name = ""

        def track_live_stuck_after_step(cur_obs: dict[str, Any], info: dict[str, Any]) -> bool:
            """Return True if the run should abort (too many stuck steps)."""
            nonlocal stuck_live_actions, prev_stuck_turn
            turn_key = (
                str(cur_obs.get("active_side", "none")),
                int(cur_obs.get("active_index", -1)),
                int(cur_obs.get("round", 0)),
            )
            if prev_stuck_turn != turn_key:
                stuck_live_actions = 0
                prev_stuck_turn = turn_key
            reason = info.get("reason")
            if reason in _LIVE_STUCK_REASONS:
                stuck_live_actions += 1
            else:
                stuck_live_actions = 0
            if args.max_stuck_live_actions > 0 and stuck_live_actions >= args.max_stuck_live_actions:
                print(
                    "circuit_breaker_stuck_live "
                    f"count={stuck_live_actions} last_reason={reason} "
                    f"active_side={cur_obs.get('active_side')} active_index={cur_obs.get('active_index')}",
                    flush=True,
                )
                return True
            return False

        for step in range(1, args.max_steps + 1):
            skip_action_this_cycle = False
            legal_for_log: list[dict[str, Any]] = []
            pre_obs_for_log = dict(obs)
            if not args.act_on_enemy_turn and obs.get("active_side") != "heroes" and not obs.get("done"):
                print(
                    "waiting_for_hero_turn "
                    f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                    flush=True,
                )
                obs = live_env.wait_for_hero_turn(max_wait=args.enemy_turn_wait, poll_interval=args.wait_poll)
                if obs.get("done"):
                    print(f"episode_end steps={step - 1} total_reward={total_reward:.3f} heroes_won={obs.get('heroes_won')}")
                    _settle_after_terminal(args)
                    break

            action_idx = -1
            if args.mode == "pass_only":
                payload = {"pass_turn": True}
            elif args.mode == "hook_only":
                # Fixed hook action for quick execute-path diagnostics.
                hero_slot = int(obs.get("active_index", 0))
                payload = {"hero_slot": hero_slot, "skill_idx": 0, "target_idx": 0}
            else:
                if agent is None or shadow_env is None:
                    raise RuntimeError(f"{args.mode} agent is not loaded")
                live_state = parse_state(_live_state_from_obs(obs))
                shadow_env.state = live_state
                _set_shadow_action_history(shadow_env, live_consecutive_moves, live_last_skill_name)
                obs_vec = _adapt_obs_for_model(shadow_env._obs(live_state), model_obs_dim)
                _, legal = _build_mask_and_legal(shadow_env, obs)
                active_side = obs.get("active_side")
                active_index = int(obs.get("active_index", -1))
                if (
                    not args.act_on_enemy_turn
                    and active_side == "heroes"
                    and _active_unit_is_stunned(obs)
                ):
                    obs, skipped = _wait_for_stunned_turn_skip(live_env, obs, args)
                    if skipped:
                        skip_action_this_cycle = True
                        payload = {}
                        action_idx = -1
                    else:
                        print(
                            "stunned_turn_skip_timeout "
                            f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                            flush=True,
                        )
                        live_state = parse_state(_live_state_from_obs(obs))
                        shadow_env.state = live_state
                        _set_shadow_action_history(shadow_env, live_consecutive_moves, live_last_skill_name)
                        obs_vec = _adapt_obs_for_model(shadow_env._obs(live_state), model_obs_dim)
                        _, legal = _build_mask_and_legal(shadow_env, obs)
                    if obs.get("done"):
                        print(f"episode_end steps={step - 1} total_reward={total_reward:.3f} heroes_won={obs.get('heroes_won')}")
                        _settle_after_terminal(args)
                        break
                    if skip_action_this_cycle:
                        continue

                if skip_action_this_cycle:
                    continue

                if obs.get("active_side") != "heroes":
                    print(
                        "stale_or_nonhero_state_after_wait "
                        f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                        flush=True,
                    )
                    obs = live_env.wait_for_hero_turn(max_wait=args.enemy_turn_wait, poll_interval=args.wait_poll)
                    continue

                obs, legal = _wait_for_hero_legal_actions(live_env, shadow_env, obs, args)
                if obs.get("done"):
                    print(f"episode_end steps={step - 1} total_reward={total_reward:.3f} heroes_won={obs.get('heroes_won')}")
                    _settle_after_terminal(args)
                    break
                if obs.get("active_side") != "heroes":
                    print(
                        "legal_action_wait_left_hero_turn "
                        f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                        flush=True,
                    )
                    continue
                live_state = parse_state(_live_state_from_obs(obs))
                shadow_env.state = live_state
                _set_shadow_action_history(shadow_env, live_consecutive_moves, live_last_skill_name)
                obs_vec = _adapt_obs_for_model(shadow_env._obs(live_state), model_obs_dim)
                pre_obs_for_log = dict(obs)

                if not legal:
                    print(
                        "safe_stop_no_legal_actions "
                        f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                        flush=True,
                    )
                    _maybe_print_terminal(
                        live_env,
                        steps=step - 1,
                        total_reward=total_reward,
                        args=args,
                        stop_reason="safe_stop_no_legal_actions",
                    )
                    return 2

                if args.log_legal_actions:
                    print(f"legal_actions={_compact_legal(shadow_env, legal)}", flush=True)
                legal_for_log = _compact_legal(shadow_env, legal)
                action_mask, payload_by_idx = _build_model_mask(
                    shadow_env,
                    legal,
                    model_action_dim,
                    live_obs=obs,
                )
                if not action_mask.any():
                    print(
                        "safe_stop_empty_action_mask "
                        f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                        flush=True,
                    )
                    _maybe_print_terminal(
                        live_env,
                        steps=step - 1,
                        total_reward=total_reward,
                        args=args,
                        stop_reason="safe_stop_empty_action_mask",
                    )
                    return 2
                action_idx, _ = agent.predict(obs_vec, action_mask, deterministic=not args.stochastic)
                payload = _decode_to_payload(shadow_env, action_idx, obs, model_action_dim, payload_by_idx)

            obs, reward, done, info = live_env.step(payload)
            total_reward += reward
            print(f"step={step} action_idx={action_idx} payload={payload} reward={reward:.3f} done={done} info={info}")
            _append_action_log(
                action_log_path,
                {
                    "step": step,
                    "mode": args.mode,
                    "turn": {
                        "round": pre_obs_for_log.get("round"),
                        "active_side": pre_obs_for_log.get("active_side"),
                        "active_index": pre_obs_for_log.get("active_index"),
                    },
                    "state": _compact_live_obs(pre_obs_for_log),
                    "legal_actions": legal_for_log,
                    "raw_legal_actions": pre_obs_for_log.get("legal_actions") or [],
                    "action_idx": int(action_idx),
                    "payload": payload,
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info,
                    "post_turn": {
                        "round": obs.get("round"),
                        "active_side": obs.get("active_side"),
                        "active_index": obs.get("active_index"),
                    },
                },
            )

            reason = info.get("reason")
            if track_live_stuck_after_step(obs, info):
                return 4
            if args.mode in model_modes and shadow_env is not None and action_idx >= 0:
                chosen_spec = shadow_env._decode_action(int(action_idx))
                if chosen_spec.kind == "move":
                    live_consecutive_moves += 1
                    live_last_skill_name = "move"
                else:
                    live_consecutive_moves = 0
                    live_last_skill_name = chosen_spec.kind

            if done:
                print(f"episode_end steps={step} total_reward={total_reward:.3f} heroes_won={info.get('heroes_won')}")
                _settle_after_terminal(args)
                break
            settle_delay = max(float(args.step_delay), float(args.post_action_settle))
            if settle_delay > 0:
                import time
                time.sleep(settle_delay)
        return 0
    finally:
        live_env.close()


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
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
from env.dd_env import ACTION_SPACE_SIZE, ACTION_TARGET_SLOTS, ENCOUNTER_HINT_DIM, MAX_SKILLS_PER_UNIT, NUM_ITEMS, DarkestDungeonEnv
from env.live.mod_state import parse_state
from dd2_env import DD2Env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trained PPO agent in live DD2.")
    p.add_argument("--model", default="runs/best/best_model.zip")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--reset-timeout", type=float, default=120.0)
    p.add_argument("--action-timeout", type=float, default=10.0)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--noop-threshold", type=int, default=2, help="Consecutive no-op acks before forced pass recovery.")
    p.add_argument("--recovery-steps", type=int, default=0, help="How many forced pass steps to run after noop threshold.")
    p.add_argument("--step-delay", type=float, default=0.25, help="Delay between live actions to avoid flooding game state.")
    p.add_argument("--mode", choices=("pass_only", "hook_only", "ppo"), default="ppo", help="Force action source for commit-path diagnostics.")
    p.add_argument("--max-no-ack-retries", type=int, default=3, help="Max empty recv cycles before step returns no_ack.")
    p.add_argument("--enemy-turn-wait", type=float, default=60.0, help="Seconds to wait for the next hero turn before sending an action.")
    p.add_argument("--wait-poll", type=float, default=0.5, help="Polling interval while waiting for enemy/animation phases to finish.")
    p.add_argument("--no-real-action-wait", type=float, default=6.0, help="Seconds to wait for non-pass live actions before stopping safely.")
    p.add_argument("--terminal-wait", type=float, default=10.0, help="Seconds to wait for battle_end before safe-stop exits.")
    p.add_argument("--terminal-settle", type=float, default=3.0, help="Seconds to keep the live connection open after battle end.")
    p.add_argument("--act-on-enemy-turn", action="store_true", help="Diagnostic override: send actions even when active_side is not heroes.")
    p.add_argument("--allow-pass-actions", action="store_true", help="Diagnostic: allow PPO/recovery to send pass_turn in live mode.")
    p.add_argument(
        "--max-stuck-live-actions",
        type=int,
        default=8,
        help="Exit after this many consecutive live steps with no battle progress (no-delta / move_unavailable / pass unavailable).",
    )
    p.add_argument("--quiet", action="store_true")
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


def _is_enemy_remnant(unit: dict[str, Any]) -> bool:
    text = " ".join(
        str(unit.get(key, ""))
        for key in ("id", "name", "archetype_id", "display_name")
    ).lower()
    return any(
        marker in text
        for marker in (
            "corpse",
            "cadaver",
            "remnant",
            "tomb",
            "grave",
            "gravestone",
            "headstone",
            "труп",
            "надгроб",
        )
    )


def _build_mask_and_legal(env: DarkestDungeonEnv, live_obs: dict[str, Any]) -> tuple[np.ndarray, list[ActionSpec]]:
    state = parse_state(_live_state_from_obs(live_obs))
    env.state = state
    legal: list[ActionSpec] = []
    raw_actions = live_obs.get("legal_actions") or []
    enemy_by_slot = {
        int(raw.get("slot", idx)): raw
        for idx, raw in enumerate(live_obs.get("enemies") or [])
    }
    remnant_legal: list[ActionSpec] = []

    for raw in raw_actions:
        if raw.get("pass_turn"):
            legal.append(ActionSpec(kind="pass"))
            continue
        if "item_id" in raw:
            target_team = str(raw.get("target_team") or "heroes")
            target_idx = int(raw.get("target_idx", 0))
            if target_team == "enemies" and _is_enemy_remnant(enemy_by_slot.get(target_idx, {})):
                remnant_legal.append(
                    ActionSpec(
                        kind="item",
                        item_id=str(raw.get("item_id") or ""),
                        target_idx=target_idx,
                        target_side=target_team,
                    )
                )
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
            target_team = str(raw.get("target_team") or "enemies")
            target_idx = int(raw.get("target_idx", 0))
            if target_team == "enemies" and _is_enemy_remnant(enemy_by_slot.get(target_idx, {})):
                remnant_legal.append(
                    ActionSpec(
                        kind="skill",
                        skill_idx=int(raw.get("skill_idx", 0)),
                        target_idx=target_idx,
                        target_side=target_team,
                    )
                )
                continue
            legal.append(
                ActionSpec(
                    kind="skill",
                    skill_idx=int(raw.get("skill_idx", 0)),
                    target_idx=target_idx,
                    target_side=target_team,
                )
            )

    if remnant_legal and not any(spec.kind != "pass" for spec in legal):
        # Corpses/remnants do not count for victory, but DD2 can still require
        # attacking them as blockers when no living enemy is targetable.
        legal = remnant_legal + [spec for spec in legal if spec.kind == "pass"]

    mask = np.zeros((ACTION_SPACE_SIZE,), dtype=bool)
    for spec in legal:
        idx = env._encode_action(spec)
        if 0 <= idx < ACTION_SPACE_SIZE:
            mask[idx] = True
    if not mask.any() and raw_actions:
        mask[-1] = True
    return mask, legal


def _legacy_action_space_size() -> int:
    return MAX_SKILLS_PER_UNIT * ACTION_TARGET_SLOTS + NUM_ITEMS * ACTION_TARGET_SLOTS + 1


def _encode_legacy_action(env: DarkestDungeonEnv, spec: ActionSpec) -> int | None:
    if spec.kind == "pass":
        return _legacy_action_space_size() - 1
    if spec.kind == "skill":
        if (spec.target_side or "enemies") != "enemies":
            return None
        skill_idx = int(spec.skill_idx or 0)
        target_idx = int(spec.target_idx or 0)
        if not (0 <= skill_idx < MAX_SKILLS_PER_UNIT and 0 <= target_idx < ACTION_TARGET_SLOTS):
            return None
        return skill_idx * ACTION_TARGET_SLOTS + target_idx
    if spec.kind == "item":
        item_ids = list(env.state.items_available.keys())[:NUM_ITEMS]
        item_idx = item_ids.index(spec.item_id) if spec.item_id in item_ids else 0
        target_idx = int(spec.target_idx or 0)
        if not (0 <= item_idx < NUM_ITEMS and 0 <= target_idx < ACTION_TARGET_SLOTS):
            return None
        return MAX_SKILLS_PER_UNIT * ACTION_TARGET_SLOTS + item_idx * ACTION_TARGET_SLOTS + target_idx
    return None


def _spec_to_payload(spec: ActionSpec, live_obs: dict[str, Any]) -> dict[str, Any]:
    if spec.kind == "pass":
        return {"pass_turn": True}
    hero_slot = int(live_obs.get("active_index", 0))
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
    return {"pass_turn": True}


def _build_model_mask(
    env: DarkestDungeonEnv,
    legal: list[ActionSpec],
    model_action_dim: int,
    allow_pass: bool,
    live_obs: dict[str, Any],
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    mask = np.zeros((model_action_dim,), dtype=bool)
    payload_by_idx: dict[int, dict[str, Any]] = {}
    ambiguous: set[int] = set()
    legal_for_policy = legal
    if not allow_pass and any(spec.kind != "pass" for spec in legal):
        legal_for_policy = [spec for spec in legal if spec.kind != "pass"]
    elif not allow_pass:
        return mask, payload_by_idx
    if model_action_dim == ACTION_SPACE_SIZE:
        for spec in legal_for_policy:
            idx = env._encode_action(spec)
            if 0 <= idx < model_action_dim:
                mask[idx] = True
                payload = _spec_to_payload(spec, live_obs)
                if idx in payload_by_idx and payload_by_idx[idx] != payload:
                    ambiguous.add(idx)
                else:
                    payload_by_idx[idx] = payload
    elif model_action_dim == _legacy_action_space_size():
        for spec in legal_for_policy:
            idx = _encode_legacy_action(env, spec)
            if idx is not None and 0 <= idx < model_action_dim:
                mask[idx] = True
                payload = _spec_to_payload(spec, live_obs)
                if idx in payload_by_idx and payload_by_idx[idx] != payload:
                    ambiguous.add(idx)
                else:
                    payload_by_idx[idx] = payload
    else:
        for spec in legal_for_policy:
            idx = env._encode_action(spec)
            if 0 <= idx < model_action_dim:
                mask[idx] = True
                payload = _spec_to_payload(spec, live_obs)
                if idx in payload_by_idx and payload_by_idx[idx] != payload:
                    ambiguous.add(idx)
                else:
                    payload_by_idx[idx] = payload
    for idx in ambiguous:
        mask[idx] = False
        payload_by_idx.pop(idx, None)
    if not mask.any() and allow_pass:
        mask[-1] = True
        payload_by_idx[int(mask.shape[0] - 1)] = {"pass_turn": True}
    return mask, payload_by_idx


def _has_real_action(legal: list[ActionSpec]) -> bool:
    return any(spec.kind != "pass" for spec in legal)


def _raw_has_move_delta(live_obs: dict[str, Any]) -> bool:
    for raw in live_obs.get("legal_actions") or []:
        if raw.get("move_delta") is not None:
            return True
    return False


def _live_non_pass_available(legal: list[ActionSpec], live_obs: dict[str, Any]) -> bool:
    """True if policy can do something other than pass: skills/items in legal, or move in plugin raw list."""
    if _has_real_action(legal):
        return True
    if _raw_has_move_delta(live_obs):
        return True
    return False


_LIVE_STUCK_REASONS = frozenset(
    {
        "move_unavailable",
        "pass_turn_unavailable",
        "no_post_state",
        "no_state_delta_after_ack",
        "selected_skill_path_no_delta",
        "no_state_delta_after_commit",
        "no_ack",
    }
)


def _legal_live_move_payload(
    live_obs: dict[str, Any],
    banned_model_idx: set[int] | None = None,
    banned_deltas: set[int] | None = None,
) -> tuple[int, dict[str, Any]] | None:
    """Use explicit move actions exposed by plugin legal_actions.

    The plugin pre-validates moves via the actor's move CombatSkill (using
    ``GetIsValidSkill`` / ``GetIsValidSkillTarget``), so any ``move_delta`` we
    receive should be safe to commit. We forward the optional ``move_skill_id``
    and target metadata so the dispatcher can short-circuit rebuilds.

    ``action_idx`` is a synthetic id (10000 + delta) for logs only; it is outside the
    PPO action space, so failed moves must be tracked via ``banned_deltas``.
    """
    banned_model_idx = banned_model_idx or set()
    banned_deltas = banned_deltas or set()
    hero_slot = int(live_obs.get("active_index", 0))
    for raw in live_obs.get("legal_actions") or []:
        if "move_delta" not in raw:
            continue
        delta = int(raw.get("move_delta", 0))
        action_idx = 10_000 + delta
        if delta in banned_deltas or action_idx in banned_model_idx:
            continue
        payload: dict[str, Any] = {"hero_slot": hero_slot, "move_delta": delta}
        for forwarded_key in ("move_skill_id", "target_idx", "target_team"):
            if forwarded_key in raw:
                payload[forwarded_key] = raw[forwarded_key]
        return action_idx, payload
    return None


def _decode_legacy_action(env: DarkestDungeonEnv, action_idx: int) -> ActionSpec:
    pass_idx = _legacy_action_space_size() - 1
    if action_idx >= pass_idx:
        return ActionSpec(kind="pass")
    skill_zone = MAX_SKILLS_PER_UNIT * ACTION_TARGET_SLOTS
    if action_idx < skill_zone:
        return ActionSpec(
            kind="skill",
            skill_idx=action_idx // ACTION_TARGET_SLOTS,
            target_idx=action_idx % ACTION_TARGET_SLOTS,
            target_side="enemies",
        )
    item_offset = action_idx - skill_zone
    item_idx = item_offset // ACTION_TARGET_SLOTS
    target_idx = item_offset % ACTION_TARGET_SLOTS
    item_ids = list(env.state.items_available.keys())[:NUM_ITEMS]
    item_id = item_ids[item_idx] if item_idx < len(item_ids) else (item_ids[0] if item_ids else "")
    return ActionSpec(kind="item", item_id=item_id, target_idx=target_idx, target_side="heroes")


def _decode_to_payload(
    env: DarkestDungeonEnv,
    action_idx: int,
    live_obs: dict[str, Any],
    model_action_dim: int,
    payload_by_idx: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if payload_by_idx and int(action_idx) in payload_by_idx:
        return dict(payload_by_idx[int(action_idx)])
    if model_action_dim == ACTION_SPACE_SIZE:
        spec = env._decode_action(int(action_idx))
    elif model_action_dim == _legacy_action_space_size():
        spec = _decode_legacy_action(env, int(action_idx))
    else:
        spec = env._decode_action(int(action_idx))
    if spec.kind == "pass":
        return {"pass_turn": True}

    hero_slot = int(live_obs.get("active_index", 0))
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
    return {"pass_turn": True}


def _adapt_obs_for_model(obs_vec: np.ndarray, expected_dim: int) -> np.ndarray:
    current_dim = int(obs_vec.shape[0])
    if current_dim == expected_dim:
        return obs_vec

    # Backward compatibility for models trained before the encounter hint
    # block was added. Relationships are the final 16 features, so remove
    # the hint block immediately before them instead of trimming the tail.
    if current_dim - expected_dim == ENCOUNTER_HINT_DIM and expected_dim >= 16:
        hint_start = current_dim - 16 - ENCOUNTER_HINT_DIM
        hint_end = current_dim - 16
        return np.concatenate([obs_vec[:hint_start], obs_vec[hint_end:]]).astype(np.float32, copy=False)

    if expected_dim < current_dim:
        return obs_vec[:expected_dim].astype(np.float32, copy=False)

    out = np.zeros((expected_dim,), dtype=np.float32)
    out[:current_dim] = obs_vec
    return out


def main() -> int:
    args = parse_args()

    shadow_env: DarkestDungeonEnv | None = None
    agent = None
    model_obs_dim = 0
    model_action_dim = 0
    if args.mode == "ppo":
        from agents.ppo_agent import PPOAgent

        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        # Shadow env is needed for PPO observation/action encoding.
        shadow_env = DarkestDungeonEnv(seed=7)
        agent = PPOAgent.load(model_path)
        model_obs_dim = int(agent.model.observation_space.shape[0])
        model_action_dim = int(agent.model.action_space.n)
        if model_obs_dim != int(shadow_env.observation_space.shape[0]):
            print(
                "model_obs_compat "
                f"model_dim={model_obs_dim} env_dim={shadow_env.observation_space.shape[0]}",
                flush=True,
            )
        if model_action_dim != int(shadow_env.action_space.n):
            print(
                "model_action_compat "
                f"model_dim={model_action_dim} env_dim={shadow_env.action_space.n}",
                flush=True,
            )

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
        consecutive_noop = 0
        recovery_left = 0
        banned_actions: dict[tuple[str, int, int], set[int]] = {}
        banned_move_deltas: dict[tuple[str, int, int], set[int]] = {}
        stuck_live_actions = 0
        prev_stuck_turn: tuple[str, int, int] | None = None

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
            selected_ban_key: tuple[str, int, int] | None = None
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
            if recovery_left > 0 and args.allow_pass_actions:
                payload = {"pass_turn": True}
                recovery_left -= 1
            elif args.mode == "pass_only":
                payload = {"pass_turn": True}
            elif args.mode == "hook_only":
                # Fixed hook action for quick execute-path diagnostics.
                hero_slot = int(obs.get("active_index", 0))
                payload = {"hero_slot": hero_slot, "skill_idx": 0, "target_idx": 0}
            else:
                if agent is None or shadow_env is None:
                    raise RuntimeError("PPO agent is not loaded")
                live_state = parse_state(_live_state_from_obs(obs))
                shadow_env.state = live_state
                obs_vec = _adapt_obs_for_model(shadow_env._obs(live_state), model_obs_dim)
                _, legal = _build_mask_and_legal(shadow_env, obs)
                if not legal or (not args.allow_pass_actions and not _has_real_action(legal)):
                    import time

                    deadline = time.time() + max(0.0, args.no_real_action_wait)
                    while (
                        time.time() < deadline
                        and (
                            not legal
                            or (
                                not args.allow_pass_actions
                                and not _live_non_pass_available(legal, obs)
                            )
                        )
                        and not obs.get("done")
                    ):
                        print(
                            "waiting_for_real_action "
                            f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                            flush=True,
                        )
                        time.sleep(max(0.05, args.wait_poll))
                        obs = live_env.refresh(max_wait=max(0.25, args.wait_poll), timeout=0.25)
                        live_state = parse_state(_live_state_from_obs(obs))
                        shadow_env.state = live_state
                        obs_vec = _adapt_obs_for_model(shadow_env._obs(live_state), model_obs_dim)
                        _, legal = _build_mask_and_legal(shadow_env, obs)

                    if not args.act_on_enemy_turn and obs.get("active_side") != "heroes" and not obs.get("done"):
                        print(
                            "defer_until_hero_turn "
                            f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}",
                            flush=True,
                        )
                        skip_action_this_cycle = True
                        payload = {}
                        action_idx = -1
                    elif not legal:
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
                    elif not args.allow_pass_actions and not _live_non_pass_available(legal, obs):
                        print(
                            "safe_stop_no_real_actions "
                            f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                            "pass_turn_disabled=true",
                            flush=True,
                        )
                        _maybe_print_terminal(
                            live_env,
                            steps=step - 1,
                            total_reward=total_reward,
                            args=args,
                            stop_reason="safe_stop_no_real_actions",
                        )
                        return 2
                    elif not args.allow_pass_actions and not _has_real_action(legal):
                        # Live DD2 can expose reposition (move_delta) while no skill/item action is legal.
                        # Use only explicit move actions from the plugin; do not probe random ally skills.
                        ban_key = (
                            str(obs.get("active_side", "none")),
                            int(obs.get("active_index", -1)),
                            int(obs.get("round", 0)),
                        )
                        banned_for_key = banned_actions.setdefault(ban_key, set())
                        banned_deltas = banned_move_deltas.setdefault(ban_key, set())
                        move_payload = _legal_live_move_payload(obs, banned_for_key, banned_deltas)
                        if move_payload is not None:
                            mv_idx, mv_payload = move_payload
                            print(
                                "force_move_no_real_actions "
                                f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                                f"action_idx={mv_idx} payload={mv_payload}",
                                flush=True,
                            )
                            obs, reward, done, info = live_env.step(mv_payload)
                            total_reward += reward
                            print(f"step={step} action_idx={mv_idx} payload={mv_payload} reward={reward:.3f} done={done} info={info}")
                            if track_live_stuck_after_step(obs, info):
                                return 4
                            if info.get("reason") in {
                                "move_unavailable",
                                "move_skill_id_missing",
                                "move_target_actor_missing",
                                "move_target_rank_invalid",
                                "selected_move_path_no_delta",
                                "selected_skill_path_no_delta",
                                "no_state_delta_after_ack",
                                "no_post_state",
                                "no_ack",
                            } or (info.get("reason", "").startswith("move_invalid")):
                                banned_for_key.add(int(mv_idx))
                                if "move_delta" in mv_payload:
                                    banned_deltas.add(int(mv_payload["move_delta"]))
                                print(
                                    "ban_move_no_delta_action "
                                    f"key={ban_key} action_idx={mv_idx} reason={info.get('reason')}",
                                    flush=True,
                                )
                                # Try the next validated move (other delta) on the next loop iteration.
                                # Do NOT probe random skills on allies - the plugin already validated moves.
                            if done:
                                print(f"episode_end steps={step} total_reward={total_reward:.3f} heroes_won={info.get('heroes_won')}")
                                _settle_after_terminal(args)
                                break
                            if args.step_delay > 0:
                                import time
                                time.sleep(args.step_delay)
                            continue
                        else:
                            print(
                                "safe_stop_moves_exhausted "
                                f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                                "pass_turn_disabled=true",
                                flush=True,
                            )
                            _maybe_print_terminal(
                                live_env,
                                steps=step - 1,
                                total_reward=total_reward,
                                args=args,
                                stop_reason="safe_stop_moves_exhausted",
                            )
                            return 2

                if skip_action_this_cycle:
                    continue

                action_mask, payload_by_idx = _build_model_mask(
                    shadow_env,
                    legal,
                    model_action_dim,
                    allow_pass=args.allow_pass_actions,
                    live_obs=obs,
                )
                ban_key = (
                    str(obs.get("active_side", "none")),
                    int(obs.get("active_index", -1)),
                    int(obs.get("round", 0)),
                )
                selected_ban_key = ban_key
                for banned_idx in banned_actions.get(ban_key, set()):
                    if 0 <= banned_idx < action_mask.shape[0]:
                        action_mask[banned_idx] = False
                if not action_mask.any():
                    print(
                        "safe_stop_empty_action_mask "
                        f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                        f"pass_actions_allowed={str(args.allow_pass_actions).lower()}",
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
                if payload.get("pass_turn") and not args.allow_pass_actions:
                    real_legal = [spec for spec in legal if spec.kind != "pass"]
                    if real_legal:
                        first_real = real_legal[0]
                        fallback_idx = (
                            _encode_legacy_action(shadow_env, first_real)
                            if model_action_dim == _legacy_action_space_size()
                            else shadow_env._encode_action(first_real)
                        )
                        if fallback_idx is not None and 0 <= fallback_idx < model_action_dim:
                            action_idx = int(fallback_idx)
                            payload = payload_by_idx.get(action_idx, _spec_to_payload(first_real, obs))
                            print(
                                "force_non_pass_fallback "
                                f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                                f"action_idx={action_idx}",
                                flush=True,
                            )
                        else:
                            print(
                                "safe_stop_pass_selected "
                                f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                                f"legal_count={len(legal)} real_legal_count={len(real_legal)} "
                                "pass_turn_disabled=true",
                                flush=True,
                            )
                            _maybe_print_terminal(
                                live_env,
                                steps=step - 1,
                                total_reward=total_reward,
                                args=args,
                                stop_reason="safe_stop_pass_selected",
                            )
                            return 2
                    else:
                        print(
                            "safe_stop_pass_selected "
                            f"active_side={obs.get('active_side')} active_index={obs.get('active_index')} "
                            f"legal_count={len(legal)} real_legal_count={len(real_legal)} "
                            "pass_turn_disabled=true",
                            flush=True,
                        )
                        _maybe_print_terminal(
                            live_env,
                            steps=step - 1,
                            total_reward=total_reward,
                            args=args,
                            stop_reason="safe_stop_pass_selected",
                        )
                        return 2

            obs, reward, done, info = live_env.step(payload)
            total_reward += reward
            print(f"step={step} action_idx={action_idx} payload={payload} reward={reward:.3f} done={done} info={info}")

            reason = info.get("reason")
            if track_live_stuck_after_step(obs, info):
                return 4

            if reason in {"no_post_state", "no_state_delta_after_ack", "selected_skill_path_no_delta", "no_state_delta_after_commit"}:
                consecutive_noop += 1
                if action_idx >= 0:
                    ban_key = selected_ban_key or (
                        str(obs.get("active_side", "none")),
                        int(obs.get("active_index", -1)),
                        int(obs.get("round", 0)),
                    )
                    banned_actions.setdefault(ban_key, set()).add(int(action_idx))
                    print(
                        "ban_no_delta_action "
                        f"key={ban_key} action_idx={action_idx} reason={reason}",
                        flush=True,
                    )
            else:
                consecutive_noop = 0

            if args.allow_pass_actions and args.recovery_steps > 0 and consecutive_noop >= args.noop_threshold:
                # Hook path likely selected but not executed by game UI/controller.
                # Force pass_turn to advance the timeline and recover from action deadlock.
                recovery_left = max(recovery_left, args.recovery_steps)
                consecutive_noop = 0

            if done:
                print(f"episode_end steps={step} total_reward={total_reward:.3f} heroes_won={info.get('heroes_won')}")
                _settle_after_terminal(args)
                break
            if args.step_delay > 0:
                import time
                time.sleep(args.step_delay)
        return 0
    finally:
        live_env.close()


if __name__ == "__main__":
    raise SystemExit(main())


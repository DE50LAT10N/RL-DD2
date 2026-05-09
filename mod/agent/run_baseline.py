# CLI entry point for running baseline policies against the live plugin.
# Helps validate IPC and action execution without loading an RL model.
# Depends on mod.agent.dd2_env and policies.py.

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from dd2_env import DD2Env
from policies import always_pass_policy, random_policy, scripted_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DD2 baseline episodes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument("--action-timeout", type=float, default=8.0)
    parser.add_argument("--policy", choices=("random", "pass", "scripted"), default="pass")
    parser.add_argument("--out", default="runs/dd2_baseline.jsonl")
    parser.add_argument("--noop-threshold", type=int, default=2, help="Consecutive no-op hook acks before forced pass recovery.")
    parser.add_argument("--recovery-steps", type=int, default=1, help="How many pass_turn actions to send after no-op threshold.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def pick_policy(name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if name == "random":
        return random_policy
    if name == "scripted":
        return scripted_policy
    return always_pass_policy


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run() -> int:
    args = parse_args()
    policy_fn = pick_policy(args.policy)
    out_path = Path(args.out)

    env = DD2Env(
        host=args.host,
        port=args.port,
        reset_timeout=args.reset_timeout,
        action_timeout=args.action_timeout,
        verbose=not args.quiet,
    )
    try:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            total_reward = 0.0
            reasons: Counter[str] = Counter()
            done = False
            heroes_won: bool | None = None
            steps_taken = 0
            consecutive_noop = 0
            recovery_left = 0

            for step_idx in range(1, args.max_steps + 1):
                if not obs.get("done") and obs.get("active_side") != "heroes":
                    try:
                        obs = env.wait_for_hero_turn(max_wait=60.0, poll_interval=0.5)
                    except TimeoutError as exc:
                        print(f"episode={episode} step={step_idx} wait_for_hero_turn timeout: {exc}")
                        break
                if obs.get("done"):
                    done = True
                    break
                if recovery_left > 0:
                    action = {"pass_turn": True}
                    recovery_left -= 1
                else:
                    action = policy_fn(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps_taken = step_idx
                reason = info.get("reason")
                if isinstance(reason, str):
                    reasons[reason] += 1
                if reason in {"no_post_state", "no_state_delta_after_ack"}:
                    consecutive_noop += 1
                else:
                    consecutive_noop = 0
                if consecutive_noop >= args.noop_threshold:
                    recovery_left = max(recovery_left, args.recovery_steps)
                    consecutive_noop = 0
                if "heroes_won" in info:
                    heroes_won = bool(info["heroes_won"])
                print(f"episode={episode} step={step_idx} action={action} reward={reward:.3f} done={done} info={info}")
                if done:
                    break

            record = {
                "episode": episode,
                "steps": steps_taken,
                "total_reward": round(total_reward, 6),
                "heroes_won": heroes_won,
                "reasons": dict(reasons),
                "policy": args.policy,
            }
            append_jsonl(out_path, record)
            print(f"episode={episode} done steps={steps_taken} total_reward={total_reward:.3f} heroes_won={heroes_won}")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())


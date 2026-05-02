from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent
from env.dd_env import DarkestDungeonEnv


HOLDOUT_ENCOUNTERS = [
    "holdout_double_ghoul",
    "holdout_double_cultist",
    "holdout_mix_wave",
    "holdout_elite_pair",
    "holdout_elite_swarm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO model on holdout encounters.")
    p.add_argument("--model", type=str, default="runs/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--seeds", type=str, default="", help="Optional comma-separated seeds, e.g. 7,17,27.")
    p.add_argument("--out-json", type=str, default="", help="Optional path to write detailed metrics JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    seed_values = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    per_seed_results: list[dict[str, object]] = []
    aggregate_summary = {"episodes": 0.0, "wins": 0.0, "reward_sum": 0.0, "steps_sum": 0.0, "survival_sum": 0.0}
    aggregate_encounters: dict[str, dict[str, float]] = {}
    normalized: dict[str, dict[str, float]] = {}

    for seed in seed_values:
        env = DarkestDungeonEnv(seed=seed)
        agent = PPOAgent.load(model_path, env=env)
        episodes = args.episodes
        wins = 0
        rewards: list[float] = []
        steps: list[int] = []
        survival_rates: list[float] = []
        by_encounter: dict[str, dict[str, float]] = {}
        for i in range(episodes):
            encounter_id = HOLDOUT_ENCOUNTERS[i % len(HOLDOUT_ENCOUNTERS)]
            obs, _ = env.reset(seed=seed + i, options={"encounter_id": encounter_id})
            done = False
            info = {}
            ep_reward = 0.0
            ep_steps = 0
            while not done:
                act, _ = agent.predict(obs, env.action_masks())
                obs, rew, term, trunc, info = env.step(act)
                ep_reward += rew
                ep_steps += 1
                done = term or trunc
            wins += 1 if info.get("heroes_won") else 0
            slot = by_encounter.setdefault(encounter_id, {"episodes": 0, "wins": 0, "reward_sum": 0.0, "steps_sum": 0.0, "survival_sum": 0.0})
            slot["episodes"] += 1
            slot["wins"] += 1 if info.get("heroes_won") else 0
            slot["reward_sum"] += ep_reward
            slot["steps_sum"] += ep_steps
            rewards.append(ep_reward)
            steps.append(ep_steps)
            alive = sum(1 for u in env.state.heroes if u.alive)
            ep_survival = alive / max(1, len(env.state.heroes))
            survival_rates.append(ep_survival)
            slot["survival_sum"] += ep_survival
        mean_reward = sum(rewards) / max(1, len(rewards))
        mean_steps = sum(steps) / max(1, len(steps))
        survival_rate = sum(survival_rates) / max(1, len(survival_rates))
        print(
            f"seed={seed} "
            f"win_rate={wins/episodes:.2%} "
            f"mean_reward={mean_reward:.3f} "
            f"mean_steps={mean_steps:.2f} "
            f"survival_rate={survival_rate:.2%}"
        )
        per_seed_results.append({
            "seed": seed,
            "summary": {
                "win_rate": wins / max(1, episodes),
                "mean_reward": mean_reward,
                "mean_steps": mean_steps,
                "survival_rate": survival_rate,
            },
        })
        aggregate_summary["episodes"] += episodes
        aggregate_summary["wins"] += wins
        aggregate_summary["reward_sum"] += sum(rewards)
        aggregate_summary["steps_sum"] += sum(steps)
        aggregate_summary["survival_sum"] += sum(survival_rates)
        for encounter_id, row in by_encounter.items():
            slot = aggregate_encounters.setdefault(encounter_id, {"episodes": 0.0, "wins": 0.0, "reward_sum": 0.0, "steps_sum": 0.0, "survival_sum": 0.0})
            slot["episodes"] += row["episodes"]
            slot["wins"] += row["wins"]
            slot["reward_sum"] += row["reward_sum"]
            slot["steps_sum"] += row["steps_sum"]
            slot["survival_sum"] += row["survival_sum"]

    total_episodes = max(1.0, aggregate_summary["episodes"])
    mean_reward = aggregate_summary["reward_sum"] / total_episodes
    mean_steps = aggregate_summary["steps_sum"] / total_episodes
    survival_rate = aggregate_summary["survival_sum"] / total_episodes
    summary = (
        f"aggregate_win_rate={aggregate_summary['wins']/total_episodes:.2%} "
        f"mean_reward={mean_reward:.3f} "
        f"mean_steps={mean_steps:.2f} "
        f"survival_rate={survival_rate:.2%}"
    )
    print(summary)
    print("encounter_breakdown:")
    for encounter_id, row in sorted(aggregate_encounters.items()):
        eps = max(1.0, row["episodes"])
        data = {
            "episodes": float(row["episodes"]),
            "win_rate": row["wins"] / eps,
            "mean_reward": row["reward_sum"] / eps,
            "mean_steps": row["steps_sum"] / eps,
            "survival_rate": row["survival_sum"] / eps,
        }
        normalized[encounter_id] = data
        print(
            f"  {encounter_id}: "
            f"win_rate={data['win_rate']:.2%} "
            f"mean_reward={data['mean_reward']:.3f} "
            f"mean_steps={data['mean_steps']:.2f} "
            f"survival_rate={data['survival_rate']:.2%}"
        )
    if args.out_json:
        out = {
            "model": str(model_path),
            "episodes_per_seed": args.episodes,
            "seeds": seed_values,
            "summary": {
                "win_rate": aggregate_summary["wins"] / total_episodes,
                "mean_reward": mean_reward,
                "mean_steps": mean_steps,
                "survival_rate": survival_rate,
            },
            "per_seed": per_seed_results,
            "encounters": normalized,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote_metrics_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

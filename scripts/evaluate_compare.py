# Compare PPO, Masked QR-DQN, and simple simulator baselines on shared holdout episodes.
# Produces one JSON/console report with identical seeds and encounters per agent.

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent
from agents.qrdqn_agent import QRDQNAgent
from env.curriculum import HOLDOUT_ENCOUNTERS
from env.data_model import ActionSpec, BattleState
from env.dd_env import DarkestDungeonEnv
from env.tactical_metrics import TacticalMetrics, attach_skill_lookup, composite_score, critical_battlefield_medicine_actions


class PolicyAdapter(Protocol):
    name: str

    def choose(self, env: DarkestDungeonEnv, obs, legal: list[ActionSpec]) -> tuple[int, ActionSpec]:
        ...


class PPOPolicy:
    def __init__(self, name: str, model_path: Path, env: DarkestDungeonEnv) -> None:
        self.name = name
        self.agent = PPOAgent.load(model_path, env=env)

    def choose(self, env: DarkestDungeonEnv, obs, legal: list[ActionSpec]) -> tuple[int, ActionSpec]:
        action_idx, _ = self.agent.predict(obs, env.action_masks())
        return _idx_to_spec(env, legal, action_idx)


class QRDQNPolicy:
    def __init__(self, name: str, model_path: Path, env: DarkestDungeonEnv, device: str = "auto") -> None:
        self.name = name
        self.agent = QRDQNAgent.load(model_path, env=env, device=device)

    def choose(self, env: DarkestDungeonEnv, obs, legal: list[ActionSpec]) -> tuple[int, ActionSpec]:
        action_idx, _ = self.agent.predict(obs, env.action_masks())
        return _idx_to_spec(env, legal, action_idx)


class RandomPolicy:
    def __init__(self, name: str, seed: int) -> None:
        self.name = name
        self.rng = random.Random(seed)

    def choose(self, env: DarkestDungeonEnv, obs, legal: list[ActionSpec]) -> tuple[int, ActionSpec]:
        chosen = self.rng.choice(legal) if legal else ActionSpec(kind="pass")
        return env._encode_action(chosen), chosen


class ScriptedPolicy:
    def __init__(self, name: str = "scripted") -> None:
        self.name = name

    def choose(self, env: DarkestDungeonEnv, obs, legal: list[ActionSpec]) -> tuple[int, ActionSpec]:
        chosen = _scripted_action(env.state, legal, env.backend)
        return env._encode_action(chosen), chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare DD2 agents on shared holdout simulator episodes.")
    p.add_argument("--ppo-model", type=str, default="", help="Optional PPO model zip to include.")
    p.add_argument("--qrdqn-model", type=str, default="", help="Optional QR-DQN model zip to include.")
    p.add_argument("--baselines", type=str, default="scripted,random", help="Comma-separated baselines: scripted,random,none.")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--seeds", type=str, default="", help="Optional comma-separated seeds, e.g. 7,17,27.")
    p.add_argument("--max-episode-steps", type=int, default=180)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--out-json", type=str, default="", help="Optional path to write comparison metrics JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_values = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    agent_specs = _agent_specs(args)
    if not agent_specs:
        raise ValueError("No agents selected. Pass --ppo-model, --qrdqn-model, or --baselines scripted,random.")

    results: dict[str, dict[str, object]] = {}
    for spec in agent_specs:
        name = str(spec["name"])
        kind = str(spec["kind"])
        env = DarkestDungeonEnv(seed=seed_values[0], max_episode_steps=args.max_episode_steps)
        policy = _make_policy(spec, env, args.device, seed_values[0])
        result = evaluate_policy_adapter(policy, env, seed_values, args.episodes)
        results[name] = result
        summary = result["summary"]
        tactical = result["tactical_metrics"]
        print(
            f"{name}: "
            f"kind={kind} "
            f"win_rate={summary['win_rate']:.2%} "
            f"mean_reward={summary['mean_reward']:.3f} "
            f"mean_steps={summary['mean_steps']:.2f} "
            f"survival_rate={summary['survival_rate']:.2%} "
            f"composite_score={summary['composite_score']:.3f} "
            f"bad_pass_rate={tactical['bad_pass_rate']:.2%}"
        )

    ranking = sorted(
        (
            {
                "agent": name,
                "win_rate": float(data["summary"]["win_rate"]),
                "mean_reward": float(data["summary"]["mean_reward"]),
                "composite_score": float(data["summary"]["composite_score"]),
            }
            for name, data in results.items()
        ),
        key=lambda row: (row["win_rate"], row["composite_score"], row["mean_reward"]),
        reverse=True,
    )
    print("ranking:")
    for i, row in enumerate(ranking, start=1):
        print(
            f"  {i}. {row['agent']}: "
            f"win_rate={row['win_rate']:.2%} "
            f"composite_score={row['composite_score']:.3f} "
            f"mean_reward={row['mean_reward']:.3f}"
        )

    if args.out_json:
        out = {
            "episodes_per_seed": args.episodes,
            "seeds": seed_values,
            "encounters": HOLDOUT_ENCOUNTERS,
            "ranking": ranking,
            "agents": results,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote_compare_json={out_path}")
    return 0


def evaluate_policy_adapter(
    policy: PolicyAdapter,
    env: DarkestDungeonEnv,
    seed_values: list[int],
    episodes_per_seed: int,
) -> dict[str, object]:
    aggregate = {"episodes": 0.0, "wins": 0.0, "reward_sum": 0.0, "steps_sum": 0.0, "survival_sum": 0.0}
    by_encounter: dict[str, dict[str, float]] = {}
    terminal_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    per_seed: list[dict[str, object]] = []
    per_episode: list[dict[str, object]] = []
    all_rewards: list[float] = []
    all_steps: list[float] = []
    all_survival: list[float] = []
    tactical = attach_skill_lookup(TacticalMetrics(), env.backend)

    for seed in seed_values:
        wins = 0
        rewards: list[float] = []
        steps: list[int] = []
        survival_rates: list[float] = []
        for i in range(episodes_per_seed):
            encounter_id = HOLDOUT_ENCOUNTERS[i % len(HOLDOUT_ENCOUNTERS)]
            obs, _ = env.reset(seed=seed + i, options={"encounter_id": encounter_id})
            tactical._backend = env.backend  # type: ignore[attr-defined]
            done = False
            info = {}
            ep_reward = 0.0
            ep_steps = 0
            ep_action_counts: Counter[str] = Counter()
            terminated = False
            truncated = False
            while not done:
                before_state = copy.deepcopy(env.state)
                legal = env.backend.legal_action_specs(env.state)
                action_idx, chosen = policy.choose(env, obs, legal)
                chosen = _ensure_legal_choice(env, legal, action_idx, chosen)
                ep_action_counts[chosen.kind] += 1
                obs, reward, term, trunc, info = env.step(action_idx)
                tactical.observe_decision(before_state, legal, chosen, env.backend.last_trace, env.state)
                ep_reward += float(reward)
                ep_steps += 1
                terminated = bool(term)
                truncated = bool(trunc)
                done = bool(term or trunc)

            won = bool(info.get("heroes_won"))
            wins += 1 if won else 0
            alive = sum(1 for unit in env.state.heroes if unit.alive)
            survival = alive / max(1, len(env.state.heroes))
            terminal = "win" if won else ("truncated" if truncated else "loss")
            slot = by_encounter.setdefault(
                encounter_id,
                {"episodes": 0.0, "wins": 0.0, "reward_sum": 0.0, "steps_sum": 0.0, "survival_sum": 0.0},
            )
            slot["episodes"] += 1
            slot["wins"] += 1 if won else 0
            slot["reward_sum"] += ep_reward
            slot["steps_sum"] += ep_steps
            slot["survival_sum"] += survival
            terminal_counts[terminal] += 1
            action_counts.update(ep_action_counts)
            rewards.append(ep_reward)
            steps.append(ep_steps)
            survival_rates.append(survival)
            all_rewards.append(ep_reward)
            all_steps.append(float(ep_steps))
            all_survival.append(survival)
            per_episode.append(
                {
                    "seed": seed,
                    "episode": i,
                    "encounter": encounter_id,
                    "won": won,
                    "reward": ep_reward,
                    "steps": ep_steps,
                    "survival_rate": survival,
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal": terminal,
                    "action_counts": dict(sorted(ep_action_counts.items())),
                }
            )

        aggregate["episodes"] += episodes_per_seed
        aggregate["wins"] += wins
        aggregate["reward_sum"] += sum(rewards)
        aggregate["steps_sum"] += sum(steps)
        aggregate["survival_sum"] += sum(survival_rates)
        per_seed.append(
            {
                "seed": seed,
                "summary": {
                    "win_rate": wins / max(1, episodes_per_seed),
                    "mean_reward": sum(rewards) / max(1, len(rewards)),
                    "mean_steps": sum(steps) / max(1, len(steps)),
                    "survival_rate": sum(survival_rates) / max(1, len(survival_rates)),
                },
            }
        )

    total = max(1.0, aggregate["episodes"])
    tactical_summary = tactical.summary()
    mean_reward = aggregate["reward_sum"] / total
    summary = {
        "episodes": aggregate["episodes"],
        "win_rate": aggregate["wins"] / total,
        "mean_reward": mean_reward,
        "mean_steps": aggregate["steps_sum"] / total,
        "survival_rate": aggregate["survival_sum"] / total,
        "reward_stats": _series_stats(all_rewards),
        "steps_stats": _series_stats(all_steps),
        "survival_stats": _series_stats(all_survival),
        "composite_score": composite_score(mean_reward, tactical_summary),
    }
    encounters = {
        encounter_id: {
            "episodes": row["episodes"],
            "win_rate": row["wins"] / max(1.0, row["episodes"]),
            "mean_reward": row["reward_sum"] / max(1.0, row["episodes"]),
            "mean_steps": row["steps_sum"] / max(1.0, row["episodes"]),
            "survival_rate": row["survival_sum"] / max(1.0, row["episodes"]),
        }
        for encounter_id, row in sorted(by_encounter.items())
    }
    return {
        "summary": summary,
        "tactical_metrics": tactical_summary,
        "per_seed": per_seed,
        "encounters": encounters,
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "episodes": per_episode,
    }


def _agent_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    if args.ppo_model:
        ppo_path = Path(args.ppo_model)
        if not ppo_path.exists():
            raise FileNotFoundError(f"PPO model not found: {ppo_path}")
        specs.append({"kind": "ppo", "name": "ppo", "path": ppo_path})
    if args.qrdqn_model:
        qrdqn_path = Path(args.qrdqn_model)
        if not qrdqn_path.exists():
            raise FileNotFoundError(f"QR-DQN model not found: {qrdqn_path}")
        specs.append({"kind": "qrdqn", "name": "qrdqn", "path": qrdqn_path})

    baselines = [x.strip().lower() for x in str(args.baselines).split(",") if x.strip()]
    if "none" in baselines:
        baselines = []
    for baseline in baselines:
        if baseline not in {"scripted", "random"}:
            raise ValueError(f"Unknown baseline: {baseline}")
        specs.append({"kind": baseline, "name": baseline})
    return specs


def _make_policy(spec: dict[str, object], env: DarkestDungeonEnv, device: str, seed: int) -> PolicyAdapter:
    kind = str(spec["kind"])
    name = str(spec["name"])
    if kind == "ppo":
        return PPOPolicy(name, Path(spec["path"]), env)
    if kind == "qrdqn":
        return QRDQNPolicy(name, Path(spec["path"]), env, device=device)
    if kind == "scripted":
        return ScriptedPolicy(name)
    if kind == "random":
        return RandomPolicy(name, seed=seed)
    raise ValueError(f"Unknown agent kind: {kind}")


def _scripted_action(state: BattleState, legal: list[ActionSpec], backend) -> ActionSpec:
    if not legal:
        return ActionSpec(kind="pass")

    critical = critical_battlefield_medicine_actions(state, legal, backend=backend)
    if critical:
        return _lowest_hp_target_action(state.heroes, critical)

    heal_actions = [a for a in legal if _skill_for_action(state, a, backend) is not None and a.target_side == "heroes"]
    useful_heals = [a for a in heal_actions if _is_heal_action(state, a, backend) and _target_hp_ratio(state.heroes, a.target_idx) <= 0.5]
    if useful_heals:
        return _lowest_hp_target_action(state.heroes, useful_heals)

    kill_actions = [a for a in legal if _is_enemy_skill_action(a) and _target_is_low_enemy(state, a)]
    if kill_actions:
        return max(kill_actions, key=lambda a: _offense_score(state, a, backend))

    attacks = [a for a in legal if _is_enemy_skill_action(a) and not _target_is_bad_remnant(state, a)]
    if attacks:
        return max(attacks, key=lambda a: (_offense_score(state, a, backend), -_target_hp_ratio(state.enemies, a.target_idx)))

    items = [a for a in legal if a.kind == "item" and a.target_side == "heroes" and _target_hp_ratio(state.heroes, a.target_idx) <= 0.5]
    if items:
        return _lowest_hp_target_action(state.heroes, items)

    non_pass = [a for a in legal if a.kind != "pass"]
    return non_pass[0] if non_pass else legal[0]


def _idx_to_spec(env: DarkestDungeonEnv, legal: list[ActionSpec], action_idx: int) -> tuple[int, ActionSpec]:
    chosen = next((a for a in legal if env._encode_action(a) == int(action_idx)), env._decode_action(int(action_idx)))
    return int(action_idx), chosen


def _ensure_legal_choice(env: DarkestDungeonEnv, legal: list[ActionSpec], action_idx: int, chosen: ActionSpec) -> ActionSpec:
    encoded = env._encode_action(chosen)
    if encoded == int(action_idx) and any(env._encode_action(a) == encoded for a in legal):
        return chosen
    return next((a for a in legal if env._encode_action(a) == int(action_idx)), chosen)


def _skill_for_action(state: BattleState, action: ActionSpec, backend):
    if action.kind != "skill" or action.skill_idx is None:
        return None
    try:
        _, _, actor = backend.get_active_unit(state)
        skills = backend._skills_for(actor)
    except Exception:
        return None
    if not (0 <= int(action.skill_idx) < len(skills)):
        return None
    return skills[int(action.skill_idx)]


def _is_heal_action(state: BattleState, action: ActionSpec, backend) -> bool:
    skill = _skill_for_action(state, action, backend)
    return bool(skill is not None and (skill.heal > 0 or skill.heal_percent > 0 or skill.heal_stress > 0 or skill.cures_tokens))


def _is_enemy_skill_action(action: ActionSpec) -> bool:
    return action.kind == "skill" and action.target_side == "enemies"


def _target_is_low_enemy(state: BattleState, action: ActionSpec) -> bool:
    target = _unit_at(state.enemies, action.target_idx)
    return bool(target is not None and target.alive and target.hp <= max(2, int(target.max_hp * 0.15)))


def _target_is_bad_remnant(state: BattleState, action: ActionSpec) -> bool:
    target = _unit_at(state.enemies, action.target_idx)
    if target is None or not getattr(target, "is_remnant", False):
        return False
    return any(e.alive and not getattr(e, "is_remnant", False) for e in state.enemies)


def _offense_score(state: BattleState, action: ActionSpec, backend) -> float:
    skill = _skill_for_action(state, action, backend)
    target = _unit_at(state.enemies, action.target_idx)
    if skill is None:
        return 0.0
    target_hp = float(getattr(target, "hp", 0) or 0)
    damage = float(skill.damage_hi + skill.dot_amount * max(1, skill.dot_duration) + skill.stress_damage)
    execute = 4.0 if getattr(skill, "execution", 0) else 0.0
    kill_bonus = 10.0 if target is not None and target.alive and target_hp <= max(2.0, damage) else 0.0
    remnant_penalty = -5.0 if target is not None and getattr(target, "is_remnant", False) else 0.0
    return damage + execute + kill_bonus + remnant_penalty


def _lowest_hp_target_action(units: list[object], actions: list[ActionSpec]) -> ActionSpec:
    return min(actions, key=lambda a: _target_hp_ratio(units, a.target_idx))


def _target_hp_ratio(units: list[object], target_idx: int | None) -> float:
    target = _unit_at(units, target_idx)
    if target is None:
        return 1.0
    return float(getattr(target, "hp", 0)) / max(1.0, float(getattr(target, "max_hp", 1)))


def _unit_at(units: list[object], idx: int | None):
    if idx is None or not (0 <= int(idx) < len(units)):
        return None
    return units[int(idx)]


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(float(x) for x in values)
    n = len(ordered)

    def percentile(frac: float) -> float:
        if n == 1:
            return ordered[0]
        pos = frac * (n - 1)
        lo = int(pos)
        hi = min(n - 1, lo + 1)
        weight = pos - lo
        return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

    mean = sum(ordered) / n
    variance = sum((x - mean) ** 2 for x in ordered) / n
    return {
        "mean": mean,
        "std": variance**0.5,
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": ordered[-1],
    }


if __name__ == "__main__":
    raise SystemExit(main())

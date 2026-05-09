# Behavior-cloning warmup from labeled live frames.
# Fine-tunes a PPO policy toward preferred live-log actions before RL continuation.
# Depends on PPOAgent, live state conversion, PyTorch, and label_live_log output.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from env.dd_env import DarkestDungeonEnv
from env.live.mod_state import parse_state
from scripts.live_ppo import _adapt_obs_for_model, _build_mask_and_legal, _live_state_from_obs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavior-cloning warmup from labeled live frames.")
    p.add_argument("--model", default="runs/best/best_model.zip")
    p.add_argument("--labels", default="runs/live_action_labels.jsonl")
    p.add_argument("--out", default="runs/bc_model.zip")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def _raw_from_compact(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("kind")
    if kind == "pass":
        return {"pass_turn": True}
    if kind == "move":
        return {"move_delta": int(action.get("move_delta", 0) or 0), "move_skill_id": "move"}
    if kind == "item":
        return {
            "item_id": action.get("item_id"),
            "skill_id": action.get("skill_id"),
            "target_idx": int(action.get("target_idx", 0) or 0),
            "target_team": str(action.get("target_side") or "heroes"),
        }
    return {
        "skill_idx": int(action.get("skill_idx", 0) or 0),
        "skill_id": action.get("skill_id"),
        "target_idx": int(action.get("target_idx", 0) or 0),
        "target_team": str(action.get("target_side") or "enemies"),
    }


def _load_samples(path: Path, env: DarkestDungeonEnv, model_obs_dim: int, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    actions: list[int] = []
    if not path.is_file():
        return np.zeros((0, model_obs_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            action_idx = row.get("preferred_action_idx")
            if action_idx is None:
                continue
            state = dict(row.get("state") or {})
            state["legal_actions"] = row.get("raw_legal_actions") or [
                _raw_from_compact(a) for a in (row.get("legal_actions") or [])
            ]
            live_state = parse_state(_live_state_from_obs(state))
            env.state = live_state
            mask, _ = _build_mask_and_legal(env, state)
            action_idx = int(action_idx)
            if not (0 <= action_idx < env.action_space.n) or not mask[action_idx]:
                continue
            obs_rows.append(_adapt_obs_for_model(env._obs(live_state), model_obs_dim))
            actions.append(action_idx)
            if max_samples > 0 and len(actions) >= max_samples:
                break
    if not actions:
        return np.zeros((0, model_obs_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(obs_rows).astype(np.float32), np.asarray(actions, dtype=np.int64)


def main() -> int:
    args = parse_args()
    env = DarkestDungeonEnv(seed=23)
    agent = PPOAgent.load(Path(args.model), env=env)
    model = agent.model
    model_obs_dim = int(model.observation_space.shape[0])
    obs, actions = _load_samples(Path(args.labels), env, model_obs_dim, args.max_samples)
    if len(actions) <= 0:
        raise ValueError(f"No labeled samples found in {args.labels}. Run scripts/label_live_log.py first.")

    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.learning_rate))
    device = policy.device
    rng = np.random.default_rng(123)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(actions))
        losses: list[float] = []
        for start in range(0, len(order), max(1, args.batch_size)):
            idx = order[start : start + max(1, args.batch_size)]
            obs_t = torch.as_tensor(obs[idx], dtype=torch.float32, device=device)
            act_t = torch.as_tensor(actions[idx], dtype=torch.long, device=device)
            _, log_prob, entropy = policy.evaluate_actions(obs_t, act_t)
            loss = -log_prob.mean() - 0.001 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"bc_epoch={epoch} samples={len(actions)} loss={sum(losses)/max(1,len(losses)):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"saved_bc_model={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

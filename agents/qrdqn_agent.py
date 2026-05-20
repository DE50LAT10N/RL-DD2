# QR-DQN model wrapper for masked inference, saving, and loading.
# Keeps the public interface close to PPOAgent while QR-DQN training support grows.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import QRDQN


class QRDQNAgent:
    def __init__(self, model: QRDQN) -> None:
        self.model = model

    @classmethod
    def load(cls, model_path: str | Path, env=None, device: str = "auto") -> "QRDQNAgent":
        return cls(QRDQN.load(str(model_path), env=env, device=device))

    def save(self, model_path: str | Path) -> None:
        self.model.save(str(model_path))

    @staticmethod
    def default_policy_kwargs() -> dict:
        return {"net_arch": [384, 384], "n_quantiles": 101}

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return mean quantile values with shape (batch, action_dim)."""
        self.model.policy.set_training_mode(False)
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with th.no_grad():
            quantiles = self.model.policy.quantile_net(obs_tensor)
            q_values = quantiles.mean(dim=1)
        return q_values.detach().cpu().numpy()

    def predict(self, observation: np.ndarray, action_masks: np.ndarray | None = None, deterministic: bool = True):
        if action_masks is None:
            action, state = self.model.predict(observation, deterministic=deterministic)
            return int(action), state

        mask = _normalize_action_mask(action_masks, int(self.model.action_space.n))
        if deterministic:
            q_values = self.q_values(observation)
            if q_values.shape[0] != 1:
                raise ValueError(f"QRDQNAgent.predict expects one observation, got batch={q_values.shape[0]}")
            masked_q = np.where(mask, q_values[0], -np.inf)
            return int(np.argmax(masked_q)), None

        action, state = self.model.predict(observation, deterministic=False)
        action_idx = int(action)
        if 0 <= action_idx < mask.shape[0] and mask[action_idx]:
            return action_idx, state
        return int(np.random.choice(np.flatnonzero(mask))), state


def _normalize_action_mask(action_masks: np.ndarray, action_dim: int) -> np.ndarray:
    mask = np.asarray(action_masks, dtype=bool)
    if mask.ndim == 2:
        if mask.shape[0] != 1:
            raise ValueError(f"QRDQNAgent.predict expects one action mask, got batch={mask.shape[0]}")
        mask = mask[0]
    if mask.ndim != 1 or mask.shape[0] != action_dim:
        raise ValueError(f"Invalid action mask shape: got {mask.shape}, expected ({action_dim},)")
    mask = mask.copy()
    if not mask.any():
        mask[-1] = True
    return mask

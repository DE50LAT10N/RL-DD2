from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO


class PPOAgent:
    def __init__(self, model: MaskablePPO) -> None:
        self.model = model

    @classmethod
    def load(cls, model_path: str | Path, env=None) -> "PPOAgent":
        return cls(MaskablePPO.load(str(model_path), env=env))

    def save(self, model_path: str | Path) -> None:
        self.model.save(str(model_path))

    @staticmethod
    def default_policy_kwargs() -> dict:
        return {"net_arch": [256, 256]}

    def predict(self, observation: np.ndarray, action_masks: np.ndarray, deterministic: bool = True):
        action, state = self.model.predict(observation, deterministic=deterministic, action_masks=action_masks)
        return int(action), state

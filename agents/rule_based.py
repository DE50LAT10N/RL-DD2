from __future__ import annotations

import numpy as np


class RuleBasedAgent:
    def predict(self, observation: np.ndarray, action_masks: np.ndarray, env=None):
        valid = np.flatnonzero(action_masks)
        if valid.size == 0:
            return 0, None
        return int(valid[0]), None

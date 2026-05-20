from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from sb3_contrib import QRDQN

from agents.qrdqn_agent import QRDQNAgent
from env.dd_env import DarkestDungeonEnv


class QRDQNAgentTest(unittest.TestCase):
    def _agent_and_env(self) -> tuple[QRDQNAgent, DarkestDungeonEnv]:
        env = DarkestDungeonEnv(seed=11)
        model = QRDQN(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [16], "n_quantiles": 7},
            learning_starts=1,
            buffer_size=10,
            verbose=0,
            device="cpu",
        )
        return QRDQNAgent(model), env

    def test_predict_respects_action_mask(self) -> None:
        agent, env = self._agent_and_env()
        obs, _ = env.reset(seed=11)
        mask = env.action_masks()

        action, _ = agent.predict(obs, mask)

        self.assertTrue(mask[action])
        self.assertEqual(agent.q_values(obs).shape, (1, env.action_space.n))

    def test_empty_mask_falls_back_to_pass(self) -> None:
        agent, env = self._agent_and_env()
        obs, _ = env.reset(seed=12)

        action, _ = agent.predict(obs, np.zeros(env.action_space.n, dtype=bool))

        self.assertEqual(action, env.action_space.n - 1)

    def test_save_load_roundtrip(self) -> None:
        agent, env = self._agent_and_env()
        tmp = Path(tempfile.gettempdir()) / "ddrl_qrdqn_agent_unittest.zip"
        try:
            agent.save(tmp)
            loaded = QRDQNAgent.load(tmp, env=env, device="cpu")
            obs, _ = env.reset(seed=13)
            action, _ = loaded.predict(obs, env.action_masks())
            self.assertTrue(env.action_masks()[action])
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

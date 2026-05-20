from __future__ import annotations

import unittest

import numpy as np

from agents.masked_qrdqn import MaskedQRDQN
from agents.masked_replay_buffer import MaskedReplayBuffer
from env.dd_env import DarkestDungeonEnv


class MaskedQRDQNTest(unittest.TestCase):
    def _model_and_env(self) -> tuple[MaskedQRDQN, DarkestDungeonEnv]:
        env = DarkestDungeonEnv(seed=41)
        model = MaskedQRDQN(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [16], "n_quantiles": 7},
            learning_starts=1,
            buffer_size=32,
            batch_size=2,
            train_freq=1,
            gradient_steps=1,
            verbose=0,
            device="cpu",
        )
        return model, env

    def test_predict_respects_mask_during_greedy_and_exploration(self) -> None:
        model, env = self._model_and_env()
        obs, _ = env.reset(seed=41)
        mask = np.zeros(env.action_space.n, dtype=bool)
        mask[4] = True

        greedy_action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        self.assertEqual(int(greedy_action[0]), 4)

        model.exploration_rate = 1.0
        explored_action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        self.assertEqual(int(explored_action[0]), 4)

    def test_warmup_sample_action_respects_mask(self) -> None:
        model, env = self._model_and_env()
        obs, _ = env.reset(seed=42)
        model._last_obs = obs.reshape(1, -1)
        mask = np.zeros((1, env.action_space.n), dtype=bool)
        mask[0, 6] = True
        model._last_action_masks = mask
        model.num_timesteps = 0

        action, buffer_action = model._sample_action(learning_starts=10, n_envs=1)

        self.assertEqual(int(action[0]), 6)
        self.assertEqual(int(buffer_action[0]), 6)

    def test_short_learn_uses_masked_replay_buffer(self) -> None:
        model, _ = self._model_and_env()

        model.learn(total_timesteps=6)

        self.assertIsInstance(model.replay_buffer, MaskedReplayBuffer)
        sample = model.replay_buffer.sample(2)
        self.assertEqual(tuple(sample.action_masks.shape), (2, model.action_space.n))
        self.assertEqual(tuple(sample.next_action_masks.shape), (2, model.action_space.n))


if __name__ == "__main__":
    unittest.main()

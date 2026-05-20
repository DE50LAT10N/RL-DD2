from __future__ import annotations

import unittest

import numpy as np

from agents.masked_replay_buffer import MaskedReplayBuffer, MaskedReplayBufferSamples
from env.dd_env import DarkestDungeonEnv


class MaskedReplayBufferTest(unittest.TestCase):
    def _transition(self):
        env = DarkestDungeonEnv(seed=21)
        obs, _ = env.reset(seed=21)
        action = int(np.flatnonzero(env.action_masks())[0])
        next_obs, reward, done, _, info = env.step(action)
        info["TimeLimit.truncated"] = False
        return env, obs, next_obs, action, reward, done, info

    def test_samples_include_explicit_masks(self) -> None:
        env, obs, next_obs, action, reward, done, info = self._transition()
        buffer = MaskedReplayBuffer(8, env.observation_space, env.action_space, device="cpu", n_envs=1)
        mask = np.zeros(env.action_space.n, dtype=bool)
        mask[3] = True
        next_mask = np.zeros(env.action_space.n, dtype=bool)
        next_mask[5] = True

        buffer.add(
            obs.reshape(1, -1),
            next_obs.reshape(1, -1),
            np.array([action]),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [info],
            action_masks=mask,
            next_action_masks=next_mask,
        )

        sample = buffer.sample(1)
        self.assertIsInstance(sample, MaskedReplayBufferSamples)
        self.assertEqual(tuple(sample.action_masks.shape), (1, env.action_space.n))
        self.assertEqual(tuple(sample.next_action_masks.shape), (1, env.action_space.n))
        self.assertTrue(bool(sample.action_masks[0, 3]))
        self.assertTrue(bool(sample.next_action_masks[0, 5]))

    def test_extracts_masks_from_infos(self) -> None:
        env, obs, next_obs, action, reward, done, info = self._transition()
        buffer = MaskedReplayBuffer(8, env.observation_space, env.action_space, device="cpu", n_envs=1)
        info["action_mask"] = np.eye(1, env.action_space.n, 7, dtype=bool)[0]
        info["next_action_mask"] = np.eye(1, env.action_space.n, 9, dtype=bool)[0]

        buffer.add(
            obs.reshape(1, -1),
            next_obs.reshape(1, -1),
            np.array([action]),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [info],
        )

        sample = buffer.sample(1)
        self.assertTrue(bool(sample.action_masks[0, 7]))
        self.assertTrue(bool(sample.next_action_masks[0, 9]))

    def test_empty_masks_fall_back_to_pass(self) -> None:
        env, obs, next_obs, action, reward, done, info = self._transition()
        buffer = MaskedReplayBuffer(8, env.observation_space, env.action_space, device="cpu", n_envs=1)

        buffer.add(
            obs.reshape(1, -1),
            next_obs.reshape(1, -1),
            np.array([action]),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [info],
            action_masks=np.zeros(env.action_space.n, dtype=bool),
            next_action_masks=np.zeros(env.action_space.n, dtype=bool),
        )

        sample = buffer.sample(1)
        self.assertTrue(bool(sample.action_masks[0, -1]))
        self.assertTrue(bool(sample.next_action_masks[0, -1]))


if __name__ == "__main__":
    unittest.main()

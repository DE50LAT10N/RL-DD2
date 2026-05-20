# Replay buffer that carries legal-action masks alongside QR-DQN transitions.
# Masked QR-DQN uses next_action_masks to avoid illegal actions in Bellman targets.

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class MaskedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor | None
    action_masks: th.Tensor
    next_action_masks: th.Tensor


class MaskedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("MaskedReplayBuffer only supports discrete action spaces.")
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.mask_dim = int(action_space.n)
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dim), dtype=bool)
        if not optimize_memory_usage:
            self.next_action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dim), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
    ) -> None:
        action_masks = self._coerce_masks(action_masks, infos, ("action_masks", "action_mask"))
        next_action_masks = self._coerce_masks(next_action_masks, infos, ("next_action_masks", "next_action_mask"))
        insert_pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.action_masks[insert_pos] = action_masks
        if self.optimize_memory_usage:
            self.action_masks[(insert_pos + 1) % self.buffer_size] = next_action_masks
        else:
            self.next_action_masks[insert_pos] = next_action_masks

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> MaskedReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            next_action_masks = self.action_masks[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            next_action_masks = self.next_action_masks[batch_inds, env_indices, :]

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            None,
            self.action_masks[batch_inds, env_indices, :],
            next_action_masks,
        )
        tensors = tuple(None if value is None else self.to_torch(value) for value in data)
        return MaskedReplayBufferSamples(*tensors)

    def _coerce_masks(
        self,
        masks: np.ndarray | None,
        infos: list[dict[str, Any]],
        info_keys: tuple[str, ...],
    ) -> np.ndarray:
        if masks is None:
            extracted = []
            for info in infos:
                value = None
                for key in info_keys:
                    if key in info:
                        value = info[key]
                        break
                if value is None:
                    extracted = []
                    break
                extracted.append(value)
            if extracted:
                masks = np.asarray(extracted, dtype=bool)

        if masks is None:
            return np.ones((self.n_envs, self.mask_dim), dtype=bool)

        arr = np.asarray(masks, dtype=bool)
        if arr.ndim == 1:
            arr = np.broadcast_to(arr, (self.n_envs, self.mask_dim))
        if arr.shape != (self.n_envs, self.mask_dim):
            raise ValueError(f"Invalid action mask shape: got {arr.shape}, expected ({self.n_envs}, {self.mask_dim})")
        arr = arr.copy()
        empty = ~arr.any(axis=1)
        if empty.any():
            arr[empty, -1] = True
        return arr

# Mask-aware QR-DQN implementation for DD2's discrete legal-action contract.
# It masks exploration, inference, and Bellman target action selection.

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import QRDQN
from sb3_contrib.qrdqn.qrdqn import quantile_huber_loss
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import TrainFrequencyUnit, should_collect_more_steps
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.vec_env import VecEnv

from agents.masked_replay_buffer import MaskedReplayBuffer


class MaskedQRDQN(QRDQN):
    def __init__(self, *args, replay_buffer_class=None, replay_buffer_kwargs=None, **kwargs):
        super().__init__(
            *args,
            replay_buffer_class=replay_buffer_class or MaskedReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            **kwargs,
        )
        self._last_action_masks: np.ndarray | None = None
        self._last_next_action_masks: np.ndarray | None = None

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        if action_masks is None:
            return super().predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

        masks = _normalize_action_masks(action_masks, int(self.action_space.n))
        if not deterministic and np.random.rand() < self.exploration_rate:
            return _sample_masked_actions(masks), state

        obs_tensor, _ = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            quantiles = self.quantile_net(obs_tensor)
            q_values = quantiles.mean(dim=1).detach().cpu().numpy()
        if q_values.shape[0] != masks.shape[0]:
            raise ValueError(f"Observation batch and action mask batch differ: {q_values.shape[0]} != {masks.shape[0]}")
        masked_q = np.where(masks, q_values, -np.inf)
        return np.argmax(masked_q, axis=1), state

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(self.action_space, spaces.Discrete):
            return super()._sample_action(learning_starts, action_noise=action_noise, n_envs=n_envs)

        masks = self._last_action_masks
        if masks is None:
            masks = np.ones((n_envs, int(self.action_space.n)), dtype=bool)
        masks = _normalize_action_masks(masks, int(self.action_space.n), n_envs=n_envs)

        if self.num_timesteps < learning_starts:
            actions = _sample_masked_actions(masks)
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            actions, _ = self.predict(self._last_obs, deterministic=False, action_masks=masks)

        return actions, actions

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer,
        action_noise: ActionNoise | None = None,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            self._last_action_masks = self._collect_action_masks(env)
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            new_obs, rewards, dones, infos = env.step(actions)
            self._last_next_action_masks = self._collect_action_masks(env)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            self._update_info_buffer(infos, dones)
            infos = self._attach_transition_masks(infos)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1
                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()
        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                next_q_values = next_quantiles.mean(dim=1, keepdim=True)
                next_masks = replay_data.next_action_masks.bool().unsqueeze(1)
                next_q_values = next_q_values.masked_fill(~next_masks, th.finfo(next_q_values.dtype).min)
                next_greedy_actions = next_q_values.argmax(dim=2, keepdim=True)
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_quantiles, 1)
                next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles

            current_quantiles = self.quantile_net(replay_data.observations)
            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _collect_action_masks(self, env: VecEnv) -> np.ndarray:
        try:
            masks = env.env_method("action_masks")
        except AttributeError:
            masks = [env.get_attr("action_masks")[0]()]
        return _normalize_action_masks(np.asarray(masks, dtype=bool), int(self.action_space.n), n_envs=env.num_envs)

    def _attach_transition_masks(self, infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._last_action_masks is None or self._last_next_action_masks is None:
            return infos
        out = []
        for idx, info in enumerate(infos):
            row = dict(info)
            row["action_mask"] = self._last_action_masks[idx]
            row["next_action_mask"] = self._last_next_action_masks[idx]
            out.append(row)
        return out


def _sample_masked_actions(masks: np.ndarray) -> np.ndarray:
    return np.asarray([np.random.choice(np.flatnonzero(mask)) for mask in masks], dtype=np.int64)


def _normalize_action_masks(masks: np.ndarray, action_dim: int, n_envs: int | None = None) -> np.ndarray:
    arr = np.asarray(masks, dtype=bool)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != action_dim:
        raise ValueError(f"Invalid action mask shape: got {arr.shape}, expected (*, {action_dim})")
    if n_envs is not None and arr.shape[0] != n_envs:
        raise ValueError(f"Invalid action mask batch: got {arr.shape[0]}, expected {n_envs}")
    arr = arr.copy()
    empty = ~arr.any(axis=1)
    if empty.any():
        arr[empty, -1] = True
    return arr

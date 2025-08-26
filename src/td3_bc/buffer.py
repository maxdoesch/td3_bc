import os
import numpy as np
import torch
import torchvision.transforms as T
import json
import minari
import logging
from typing import Dict, Tuple, Optional, Union

import td3_bc.utils as utils


def normalize(array: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-3):
    return (array - mean) / (std + eps)


class ReplayBuffer:
    def __init__(
        self,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        max_size: int = int(1e6),
        device: Optional[str] = None,
        augmentations: bool = True,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim

        self.is_image_obs = len(self.obs_shape) >= 2
        self.max_size = (
            int(1e5) if self.is_image_obs else max_size
        )  # hack to avoid MemoryError with large image buffers

        self.frame_stack = 1
        if self.is_image_obs and len(self.obs_shape) >= 4:
            self.frame_stack = self.obs_shape[0]
            self.obs_shape = self.obs_shape[1:] if self.frame_stack == 1 else self.obs_shape

        obs_dtype = torch.uint8 if self.is_image_obs else torch.float32
        self.obs = torch.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype, device="cpu")
        self.next_obs = torch.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype, device="cpu")
        self.action = torch.zeros((self.max_size, action_dim), dtype=torch.float32, device="cpu")
        self.reward = torch.zeros((self.max_size, 1), dtype=torch.float32, device="cpu")
        self.not_done = torch.zeros((self.max_size, 1), dtype=torch.float32, device="cpu")

        self.obs_mean = (
            torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if self.is_image_obs
            else torch.zeros(self.obs_shape, dtype=torch.float32, device=self.device)
        )
        self.obs_std = (
            torch.tensor(255.0, dtype=torch.float32, device=self.device)
            if self.is_image_obs
            else torch.ones(self.obs_shape, dtype=torch.float32, device=self.device)
        )

        self._staging = None  # for async H2D transfers

        if self.is_image_obs and augmentations:
            self.augmentations = T.Compose(
                [
                    T.RandomCrop(self.obs_shape[-2:], padding=4, padding_mode="constant"),
                ]
            )
        else:
            self.augmentations = None

    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, done: np.ndarray):
        """
        Add transitions to the replay buffer in a vectorized way.

        Args:
            obs (np.ndarray): Unnormalized current obs, shape (n_env, *obs_shape) or (*obs_shape,)
            action (np.ndarray): Action taken, shape (n_env, action_dim) or (action_dim,)
            next_obs (np.ndarray): Unnormalize next obs, shape (n_env, *obs_shape) or (*obs_shape,)
            reward (np.ndarray): Reward received, shape (n_env, 1) or (1,)
            done (np.ndarray): Done flag, shape (n_env, 1) or (1,)
        """
        # Convert to batch if single transition
        obs = np.expand_dims(obs, 0) if obs.ndim == len(self.obs_shape) else obs
        next_obs = np.expand_dims(next_obs, 0) if next_obs.ndim == len(self.obs_shape) else next_obs
        action = np.expand_dims(action, 0) if action.ndim == 1 else action
        reward = np.expand_dims(reward, 1) if reward.ndim == 1 else reward
        done = np.expand_dims(done, 1) if done.ndim == 1 else done

        assert obs.shape[0] == action.shape[0] == next_obs.shape[0] == reward.shape[0] == done.shape[0], (
            "All inputs must have the same first dimension (number of environments)."
        )

        n_env = obs.shape[0]

        indices = torch.arange(self.ptr, self.ptr + n_env) % self.max_size

        obs_dtype = torch.uint8 if self.is_image_obs else torch.float32
        self.obs[indices] = torch.tensor(obs, dtype=obs_dtype)
        self.next_obs[indices] = torch.tensor(next_obs, dtype=obs_dtype)
        self.action[indices] = torch.tensor(action, dtype=torch.float32)
        self.reward[indices] = torch.tensor(reward, dtype=torch.float32)
        self.not_done[indices] = 1 - torch.tensor(done, dtype=torch.float32)

        self.ptr = (self.ptr + n_env) % self.max_size
        self.size = min(self.size + n_env, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the following keys:
                - "obs": Tensor of shape (batch_size, *obs_shape) with normalized observations.
                - "action": Tensor of shape (batch_size, action_dim) with actions taken.
                - "next_obs": Tensor of shape (batch_size, *obs_shape) with normalized next observations.
                - "reward": Tensor of shape (batch_size) with rewards received.
                - "not_done": Tensor of shape (batch_size) indicating whether the episode has not ended.
        """
        idx = torch.randint(0, self.size, size=(batch_size,))

        obs_cpu = self.obs.index_select(0, idx)  # uint8 on CPU
        next_obs_cpu = self.next_obs.index_select(0, idx)
        act_cpu = self.action.index_select(0, idx)
        rew_cpu = self.reward.index_select(0, idx)
        nd_cpu = self.not_done.index_select(0, idx)

        if self._staging is None:
            obs_dtype = torch.uint8 if self.is_image_obs else torch.float32
            self._staging = {
                "obs": torch.empty((batch_size, *self.obs_shape), dtype=obs_dtype, pin_memory=True),
                "next_obs": torch.empty((batch_size, *self.obs_shape), dtype=obs_dtype, pin_memory=True),
                "action": torch.empty((batch_size, self.action_dim), dtype=torch.float32, pin_memory=True),
                "reward": torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True),
                "not_done": torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True),
            }

        self._staging["obs"].copy_(obs_cpu, non_blocking=False)
        self._staging["next_obs"].copy_(next_obs_cpu, non_blocking=False)
        self._staging["action"].copy_(act_cpu, non_blocking=False)
        self._staging["reward"].copy_(rew_cpu, non_blocking=False)
        self._staging["not_done"].copy_(nd_cpu, non_blocking=False)

        # Async H2D; cast images to float on GPU
        obs = self._staging["obs"].to(self.device, non_blocking=True)
        next_obs = self._staging["next_obs"].to(self.device, non_blocking=True)
        if self.is_image_obs:
            obs = obs.float()
            next_obs = next_obs.float()

        action = self._staging["action"].to(self.device, non_blocking=True)
        reward = self._staging["reward"].to(self.device, non_blocking=True)
        not_done = self._staging["not_done"].to(self.device, non_blocking=True)

        if self.is_image_obs and self.augmentations:
            obs = self.augmentations(obs)
            next_obs = self.augmentations(next_obs)

        obs_norm = normalize(obs, self.obs_mean, self.obs_std)
        next_obs_norm = normalize(next_obs, self.obs_mean, self.obs_std)

        return {
            "obs": obs_norm,
            "action": action,
            "next_obs": next_obs_norm,
            "reward": reward,
            "not_done": not_done,
        }

    def _get_stacked_observations(self, obs: np.array) -> Tuple[np.ndarray, np.ndarray]:
        if self.frame_stack > 1:
            obs_padded = np.concatenate(
                [np.repeat(obs[:1], self.frame_stack - 1, axis=0), obs], axis=0
            )  # -> (T+1+F-1, C, H, W) = (T+F, C, H, W)

            obs = np.lib.stride_tricks.sliding_window_view(
                obs_padded[:-1], window_shape=self.frame_stack, axis=0
            )  # (T, F, C, H, W)
            obs = np.moveaxis(obs, -1, 1)
            next_obs = np.lib.stride_tricks.sliding_window_view(
                obs_padded[1:], window_shape=self.frame_stack, axis=0
            )  # (T, F, C, H, W)
            next_obs = np.moveaxis(next_obs, -1, 1)
        else:
            next_obs = obs[1:]  # (T, C, H, W)
            obs = obs[:-1]  # (T, C, H, W)

        return obs, next_obs

    def convert_dict(self, dict_dataset):
        """
        Populate the replay buffer with transitions from a dictionary dataset.

        Args:
            dict_dataset (dict): A dictionary containing episode data with the following keys:
                - "obs" (list of np.ndarray): Observations for each episode, where each element is an array of shape (episode_length, state_dim).
                - "acts" (list of np.ndarray): Actions for each episode, where each element is an array of shape (episode_length, action_dim).
                - "rews" (list of np.ndarray): Rewards for each episode, where each element is an array of shape (episode_length,).
        """

        for episode in range(len(dict_dataset["acts"])):
            obs = np.array(dict_dataset["obs"][episode])
            acts = np.array(dict_dataset["acts"][episode])
            rews = np.array(dict_dataset["rews"][episode])
            done = np.concatenate(
                [
                    np.zeros_like(dict_dataset["rews"][episode][:-1]),
                    np.ones_like(dict_dataset["rews"][episode][-1:]),
                ]
            )

            obs, next_obs = self._get_stacked_observations(obs)

            transition = {"obs": obs, "action": acts, "next_obs": next_obs, "reward": rews, "done": done}

            self.add(**transition)

        self.obs = self.obs[: self.size]
        self.action = self.action[: self.size]
        self.reward = self.reward[: self.size]
        self.next_obs = self.next_obs[: self.size]
        self.not_done = self.not_done[: self.size]

    def convert_minari(self, dataset: minari.MinariDataset):
        # assert dataset.observation_space.shape == self.obs_shape or dataset.observation_space.shape[::-1] == self.obs_shape, "Observation dimension mismatch."

        assert dataset.action_space.shape[0] == self.action_dim, "Action dimension mismatch."

        for episode in dataset.iterate_episodes():
            observations = utils.uncombine_stacked_frames(episode.observations)
            obs, next_obs = self._get_stacked_observations(observations)
            transition = {
                "obs": obs,
                "action": episode.actions,
                "next_obs": next_obs,
                "reward": episode.rewards,
                "done": episode.terminations,
            }
            self.add(**transition)

        self.obs = self.obs[: self.size]
        self.action = self.action[: self.size]
        self.reward = self.reward[: self.size]
        self.next_obs = self.next_obs[: self.size]
        self.not_done = self.not_done[: self.size]

    def save_statistics(self, stats_path: str):
        """
        Save dataset statistics (mean and standard deviation of observations) to a JSON file.

        Args:
            stats_path (str): Directory path or file path to save the statistics JSON file.
        """

        if not stats_path.endswith(".json"):
            stats_path = os.path.join(stats_path, "dataset_statistics.json")

        stats = {
            "obs_mean": self.obs_mean.cpu().tolist(),
            "obs_std": self.obs_std.cpu().tolist(),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

    def load_statistics(self, stats_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset statistics (mean and standard deviation of observations) from a JSON file
        and set them using `set_dataset_statistics`.

        Args:
            stats_path (str): Directory path or file path to the JSON file containing statistics.
        """
        if not stats_path.endswith(".json"):
            stats_path = os.path.join(stats_path, "dataset_statistics.json")

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            obs_mean = np.array(stats["obs_mean"])
            obs_std = np.array(stats["obs_std"])
            self.set_dataset_statistics(obs_mean, obs_std)
        else:
            logging.warning(f"Dataset statistics not found at {stats_path}. Replay buffer will not be normalized.")
            obs_mean = self.obs_mean.cpu().numpy()
            obs_std = self.obs_std.cpu().numpy()

        return obs_mean, obs_std

    def compute_dataset_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and standard deviation of the observation dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - obs_mean (np.ndarray): The mean of the observations.
                - obs_std (np.ndarray): The standard deviation of the observations.
        """

        if self.is_image_obs:
            obs_mean = torch.tensor(0, dtype=torch.float32)
            obs_std = torch.tensor(255.0, dtype=torch.float32)
        else:
            obs_mean = torch.mean(self.obs[: self.size], axis=0, keepdims=False)
            obs_std = torch.std(self.obs[: self.size], axis=0, keepdims=False)

        return obs_mean.numpy(), obs_std.numpy()

    def set_dataset_statistics(self, obs_mean: np.ndarray, obs_std: np.ndarray):
        """
        Set the mean and standard deviation for the dataset.

        Args:
            obs_mean (np.ndarray): The mean to use for normalization.
            obs_std (np.ndarray): The standard deviation to use for normalization.
        """
        self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.obs_std = torch.tensor(obs_std, dtype=torch.float32, device=self.device)

    def get_dataset_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current dataset statistics (mean and standard deviation).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - obs_mean (np.ndarray): The mean of the observations.
                - obs_std (np.ndarray): The standard deviation of the observations.
        """
        return self.obs_mean.cpu().numpy(), self.obs_std.cpu().numpy()


if __name__ == "__main__":
    # Example usage
    obs_shape = 3
    action_dim = 4
    max_size = int(1e6)
    n_env = 5

    buffer = ReplayBuffer(obs_shape, action_dim, max_size)

    # Simulate adding transitions
    transition = {
        "obs": np.random.rand(n_env, obs_shape),
        "action": np.random.rand(n_env, action_dim),
        "next_obs": np.random.rand(n_env, obs_shape),
        "reward": np.random.rand(n_env, 1),
        "done": np.random.randint(0, 2, size=(n_env, 1)),
    }

    buffer.add(**transition)
    print("Buffer size after adding transitions:", buffer.size)
    print("Buffer pointer after adding transitions:", buffer.ptr)

    # Sample a batch
    batch = buffer.sample(batch_size=2)
    print("Sampled batch:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    # Convert a dictionary dataset
    episodes = 3
    episode_length = 1000
    dict_dataset = {
        "obs": [np.random.randn(episode_length + 1, obs_shape) for _ in range(episodes)],
        "acts": [np.random.randn(episode_length, action_dim) for _ in range(episodes)],
        "rews": [np.random.randn(episode_length) for _ in range(episodes)],
        "dones": [np.random.randint(0, 2, size=episode_length) for _ in range(episodes)],
    }
    buffer.convert_dict(dict_dataset)
    print("Buffer size after converting dictionary dataset:", buffer.size)
    print("Buffer pointer after converting dictionary dataset:", buffer.ptr)

    # Normalize obs
    mean, std = buffer.compute_dataset_statistics()
    buffer.set_dataset_statistics(mean, std)
    print("Mean:", mean)
    print("Std:", std)

    print("---------------------------------------------------------")

    obs_shape = (32, 32, 3)
    buffer = ReplayBuffer(obs_shape, action_dim, max_size=int(1e6))
    print("Replay buffer initialized with obs_shape:", buffer.obs_shape, "and action_dim:", buffer.action_dim)

    episode_length = 100
    dict_dataset = {
        "obs": [np.random.randn(episode_length + 1, *obs_shape) for _ in range(episodes)],
        "acts": [np.random.randn(episode_length, action_dim) for _ in range(episodes)],
        "rews": [np.random.randn(episode_length) for _ in range(episodes)],
        "dones": [np.random.randint(0, 2, size=episode_length) for _ in range(episodes)],
    }
    buffer.convert_dict(dict_dataset)
    print("Buffer size after converting dictionary dataset with complex shapes:", buffer.size)
    print("Buffer pointer after converting dictionary dataset with complex shapes:", buffer.ptr)

    mean, std = buffer.compute_dataset_statistics()
    buffer.set_dataset_statistics(mean, std)

    batch = buffer.sample(batch_size=2)
    print("Sampled batch:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    print("---------------------------------------------------------")

    frame_stack = 3
    obs_shape = (3, 64, 64)
    buffer = ReplayBuffer((frame_stack, *obs_shape), action_dim, max_size=int(1e5), augmentations=False)
    print("Replay buffer initialized with obs_shape:", buffer.obs_shape, "and action_dim:", buffer.action_dim)

    episode_length = 1000
    dict_dataset = {
        "obs": [
            np.broadcast_to(
                np.arange(episode_length + 1).reshape(-1, *([1] * len(obs_shape))), (episode_length + 1, *obs_shape)
            )
            for _ in range(episodes)
        ],
        "acts": [np.random.randn(episode_length, action_dim) for _ in range(episodes)],
        "rews": [np.random.randn(episode_length) for _ in range(episodes)],
        "dones": [np.random.randint(0, 2, size=episode_length) for _ in range(episodes)],
    }
    buffer.convert_dict(dict_dataset)
    print("Buffer size after converting dictionary dataset with complex shapes:", buffer.size)
    print("Buffer pointer after converting dictionary dataset with complex shapes:", buffer.ptr)

    mean, std = buffer.compute_dataset_statistics()
    buffer.set_dataset_statistics(mean, std)

    batch = buffer.sample(batch_size=256)
    print("Sampled batch:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    assert torch.allclose(255 * batch["obs"][0] + 1, 255 * batch["next_obs"][0]), (
        "Observation and next observation do not match"
    )
    assert torch.allclose(
        torch.stack([255 * batch["obs"][0, 0] + i for i in range(frame_stack)]), 255 * batch["obs"][0]
    ), "Stacked observations do not match expected values"

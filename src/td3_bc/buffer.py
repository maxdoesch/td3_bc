import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import IterableDataset, DataLoader
import json
import minari
import logging
from typing import Dict, Tuple, Optional, Union
from abc import abstractmethod, ABC

import td3_bc.utils as utils


def normalize(array: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-3):
    return (array - mean) / (std + eps)

class ReplayBuffer(ABC):
    def __init__(
        self,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device


        self.obs_mean = 0.0
        self.obs_std = 1.0

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def load_minari(self, dataset: minari.MinariDataset):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        pass

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

class ReplayBufferState(ReplayBuffer):
    def __init__(
        self,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        max_size: int = int(1e6),
        device: Optional[str] = None,
    ):
        super().__init__(obs_shape, action_dim, device)

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim

        self.obs = torch.zeros((self.max_size,) + self.obs_shape, dtype=torch.float32, device="cpu")
        self.next_obs = torch.zeros((self.max_size,) + self.obs_shape, dtype=torch.float32, device="cpu")
        self.action = torch.zeros((self.max_size, action_dim), dtype=torch.float32, device="cpu")
        self.reward = torch.zeros((self.max_size, 1), dtype=torch.float32, device="cpu")
        self.not_done = torch.zeros((self.max_size, 1), dtype=torch.float32, device="cpu")

        self.obs_mean = torch.zeros(self.obs_shape, dtype=torch.float32, device=self.device)
        self.obs_std = torch.ones(self.obs_shape, dtype=torch.float32, device=self.device)

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

        self.obs[indices] = torch.tensor(obs, dtype=torch.float32)
        self.next_obs[indices] = torch.tensor(next_obs, dtype=torch.float32)
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

        obs = self.obs.index_select(0, idx).to(self.device)  # uint8 on CPU
        next_obs = self.next_obs.index_select(0, idx).to(self.device)
        actions = self.action.index_select(0, idx).to(self.device)
        rewards = self.reward.index_select(0, idx).to(self.device)
        not_dones = self.not_done.index_select(0, idx).to(self.device)

        obs_norm = normalize(obs, self.obs_mean, self.obs_std)
        next_obs_norm = normalize(next_obs, self.obs_mean, self.obs_std)

        return {
            "obs": obs_norm,
            "action": actions,
            "next_obs": next_obs_norm,
            "reward": rewards,
            "not_done": not_dones,
        }

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
            obs = np.array(dict_dataset["obs"][episode][:-1])
            next_obs = np.array(dict_dataset["obs"][episode][1:])
            acts = np.array(dict_dataset["acts"][episode])
            rews = np.array(dict_dataset["rews"][episode])
            done = np.concatenate(
                [
                    np.zeros_like(dict_dataset["rews"][episode][:-1]),
                    np.ones_like(dict_dataset["rews"][episode][-1:]),
                ]
            )

            transition = {"obs": obs, "action": acts, "next_obs": next_obs, "reward": rews, "done": done}

            self.add(**transition)

        self.obs = self.obs[: self.size]
        self.action = self.action[: self.size]
        self.reward = self.reward[: self.size]
        self.next_obs = self.next_obs[: self.size]
        self.not_done = self.not_done[: self.size]

    def load_minari(self, dataset: minari.MinariDataset):
        assert dataset.action_space.shape[0] == self.action_dim, "Action dimension mismatch."

        for episode in dataset.iterate_episodes():
            transition = {
                "obs": episode.observations[:-1],
                "action": episode.actions,
                "next_obs": episode.observations[1:],
                "reward": episode.rewards,
                "done": episode.terminations | episode.truncations,
            }
            self.add(**transition)

        self.obs = self.obs[: self.size]
        self.action = self.action[: self.size]
        self.reward = self.reward[: self.size]
        self.next_obs = self.next_obs[: self.size]
        self.not_done = self.not_done[: self.size]
    
class ImageDataset(IterableDataset):
    def __init__(self, 
                minari_dataset: minari.MinariDataset,
                frame_stack: int = 1,
                sample_fraction: float = 0.4, 
                prefetch_episodes: int = 10, 
                ringbuffer_size: int = 2e4
                ):
        
        self.minari_dataset = minari_dataset
        self.frame_stack = frame_stack
        self.prefetch_episodes = prefetch_episodes
        self.ringbuffer_size = int(ringbuffer_size)
        self.sample_fraction = sample_fraction

        self.obs_buffer = None
        self.next_obs_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.not_done_buffer = None
        self.ptr = 0
        self.size = 0

    def _get_stacked_observations(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def _fetch_n_episodes(self, n: int) -> Tuple:
        ep_indices = np.random.randint(0, len(self.minari_dataset), size=n)

        episodes = self.minari_dataset.iterate_episodes(episode_indices=ep_indices)
        obs = []
        next_obs = []
        actions = []
        rewards = []
        not_dones = []

        for ep in episodes:
            observations = utils.uncombine_stacked_frames(ep.observations) #N, R, C, H, W
            observations, next_observations = self._get_stacked_observations(observations) #N-1, F, R, C, H, W
            actions_ep = np.asarray(ep.actions) #N-1, A
            rewards_ep = np.asarray(ep.rewards) #N-1,
            not_dones_ep = 1 - np.asarray(ep.terminations | ep.truncations) #N-1,

            # observations:      (N-1, F, R, C, H, W)
            # next_observations: (N-1, F, R, C, H, W)

            if self.frame_stack > 1:
                valid_obs_per_frame  = (observations != 0).any(axis=(2, 3, 4, 5))         # (N-1, F)
                valid_next_per_frame = (next_observations != 0).any(axis=(2, 3, 4, 5))    # (N-1, F)

                mask_obs  = valid_obs_per_frame.all(axis=1)                                # (N-1,)
                mask_next = valid_next_per_frame.all(axis=1)                               # (N-1,)

                mask = mask_obs & mask_next                                                # (N-1,)
            else:
                mask_obs  = (observations != 0).any(axis=(1, 2, 3, 4))                     # (N-1,)
                mask_next = (next_observations != 0).any(axis=(1, 2, 3, 4))                # (N-1,)
                mask = mask_obs & mask_next
            
            obs.append(observations[mask])
            next_obs.append(next_observations[mask])
            actions.append(actions_ep[mask])
            rewards.append(rewards_ep[mask])
            not_dones.append(not_dones_ep[mask])

        obs = np.concatenate(obs, axis=0)
        next_obs = np.concatenate(next_obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        not_dones = np.concatenate(not_dones, axis=0)

        rewards = np.expand_dims(rewards, 1) if rewards.ndim == 1 else rewards
        not_dones = np.expand_dims(not_dones, 1) if not_dones.ndim == 1 else not_dones

        return obs, next_obs, actions, rewards, not_dones

    def __iter__(self):
        while True:

            obs, next_obs, actions, rewards, not_dones = self._fetch_n_episodes(self.prefetch_episodes)

            if self.obs_buffer is None:
                obs_shape = obs.shape[1:]
                action_shape = actions.shape[1:]

                self.obs_buffer = np.zeros((self.ringbuffer_size, *obs_shape), dtype=obs.dtype)
                self.next_obs_buffer = np.zeros((self.ringbuffer_size, *obs_shape), dtype=obs.dtype)
                self.action_buffer = np.zeros((self.ringbuffer_size, *action_shape), dtype=actions.dtype)
                self.reward_buffer =  np.zeros((self.ringbuffer_size, 1), dtype=rewards.dtype)
                self.not_done_buffer = np.zeros((self.ringbuffer_size, 1), dtype=not_dones.dtype)

            n_samples = obs.shape[0]
            end = (self.ptr + n_samples) % self.ringbuffer_size
            first = min(n_samples, self.ringbuffer_size - self.ptr)
            sl1 = slice(self.ptr, self.ptr + first)
            sl2 = slice(0, max(0, n_samples - first))

            self.obs_buffer[sl1] = obs[:first]
            self.next_obs_buffer[sl1] = next_obs[:first]
            self.action_buffer[sl1] = actions[:first]
            self.reward_buffer[sl1] = rewards[:first]
            self.not_done_buffer[sl1] = not_dones[:first]
            # wrap chunk if needed
            if n_samples > first:
                self.obs_buffer[sl2] = obs[first:]
                self.next_obs_buffer[sl2] = next_obs[first:]
                self.action_buffer[sl2] = actions[first:]
                self.reward_buffer[sl2] = rewards[first:]
                self.not_done_buffer[sl2] = not_dones[first:]
            self.ptr = end
            self.size = min(self.size + n_samples, self.ringbuffer_size)

            step_indices = np.random.choice(self.size, int(self.sample_fraction * self.size), replace=False)

            for t in step_indices:
                yield {
                    'obs': self.obs_buffer[t],
                    'action': self.action_buffer[t],
                    'reward': self.reward_buffer[t],
                    'not_done': self.not_done_buffer[t],
                    'next_obs': self.next_obs_buffer[t],
            }

class ReplayBufferImage(ReplayBuffer):
    def __init__(
        self,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        frame_stack: int = 1,
        n_workers: int = 4,
        device: Optional[str] = None
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, device=device)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim
        self.frame_stack = frame_stack

        self.n_workers = n_workers

        self.augmentations = T.Compose(
            [
                utils.RandomCropDual(self.obs_shape[-2:], padding=4, padding_mode="constant"),
                utils.RandomPartialRPermutation() if len(self.obs_shape) > 4 else T.Lambda(lambda x: x),
                #utils.ColorJitterDual(brightness=0.4, contrast=0.4, saturation=0.4) if len(self.obs_shape) > 4 else T.Lambda(lambda x: x),
                #utils.RandomErasingDual(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0) if len(self.obs_shape) > 4 else T.Lambda(lambda x: x)
            ]
        )

        self.obs_mean = 0.0
        self.obs_std = 255.0

        self.dataset = None
        self.data_loader = None
        self.iter_data_loader = None

    def load_minari(self, dataset: minari.MinariDataset):
        self.dataset = ImageDataset(dataset, frame_stack=self.frame_stack)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.data_loader is None:
            self.data_loader = DataLoader(self.dataset, batch_size=batch_size, num_workers=self.n_workers, pin_memory=True, persistent_workers=True)
            self.iter_data_loader = iter(self.data_loader)

        batch = next(self.iter_data_loader)

        obs = batch['obs'].to(self.device, non_blocking=True).float()
        next_obs = batch['next_obs'].to(self.device, non_blocking=True).float()
        action = batch['action'].to(self.device, non_blocking=True).float()
        reward = batch['reward'].to(self.device, non_blocking=True).float()
        not_done = batch['not_done'].to(self.device, non_blocking=True).float()

        obs = normalize(obs, self.obs_mean, self.obs_std)
        next_obs = normalize(next_obs, self.obs_mean, self.obs_std)

        obs, next_obs = self.augmentations((obs, next_obs))

        return {
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward,
            "not_done": not_done,
        }
    
    def compute_dataset_statistics(self):
        return self.obs_mean, self.obs_std

if __name__ == "__main__":
    # Example usage
    obs_shape = 3
    action_dim = 4
    max_size = int(1e6)
    n_env = 5

    buffer = ReplayBufferState(obs_shape, action_dim, max_size)

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

    obs_shape = (3, 32, 32)
    buffer = ReplayBufferState(obs_shape, action_dim)
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
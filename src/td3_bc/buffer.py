import os
import numpy as np
import torch
import json
import minari
import logging
from typing import Dict, Tuple, Optional, Union


def normalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-3):
    return (array - mean) / (std + eps)


class ReplayBuffer:
    def __init__(
            self, 
            obs_shape: Union[int, Tuple[int, ...]], 
            action_dim: int, 
            max_size: int = int(1e7), 
            device: Optional[str] = None
            ):
        
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        self.action_dim = action_dim

        self.obs = np.zeros((max_size,) + self.obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((max_size,) + self.obs_shape, dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.obs_mean = np.zeros(self.obs_shape, dtype=np.float32)
        self.obs_std = np.ones(self.obs_shape, dtype=np.float32)

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

        obs = normalize(obs, self.obs_mean, self.obs_std)
        next_obs = normalize(next_obs, self.obs_mean, self.obs_std)

        n_env = obs.shape[0]

        indices = np.arange(self.ptr, self.ptr + n_env) % self.max_size

        self.obs[indices] = obs
        self.action[indices] = action
        self.next_obs[indices] = next_obs
        self.reward[indices] = reward
        self.not_done[indices] = 1 - done

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
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32).to(self.device),
            "action": torch.tensor(self.action[idx], dtype=torch.float32).to(self.device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32).to(self.device),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32).to(self.device),
            "not_done": torch.tensor(self.not_done[idx], dtype=torch.float32).to(self.device),
        }

#    def convert_dict(self, dict_dataset):
#        """
#        Populate the replay buffer with transitions from a dictionary dataset.
#        """
#        for episode in range(len(dict_dataset["obs"])):
#            transition = {
#                "obs": dict_dataset["obs"][episode],
#                "action": dict_dataset["acts"][episode],
#                "next_obs": dict_dataset["next_obs"][episode],
#                "reward": dict_dataset["rews"][episode],
#                "done": dict_dataset["dones"][episode],
#            }
#
#            self.add(**transition)
#
#        self.obs = self.obs[: self.size]
#        self.action = self.action[: self.size]
#        self.reward = self.reward[: self.size]
#        self.next_obs = self.next_obs[: self.size]
#        self.not_done = self.not_done[: self.size]


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
            transition = {
                "obs": np.array(dict_dataset["obs"][episode][:-1]),
                "action": np.array(dict_dataset["acts"][episode]),
                "next_obs": np.array(dict_dataset["obs"][episode][1:]),
                "reward": np.array(dict_dataset["rews"][episode]),
                "done": np.concatenate(
                    [
                        np.zeros_like(dict_dataset["rews"][episode][:-1]),
                        np.ones_like(dict_dataset["rews"][episode][-1:]),
                    ]
                ),
            }

            self.add(**transition)

        self.obs = self.obs[: self.size]
        self.action = self.action[: self.size]
        self.reward = self.reward[: self.size]
        self.next_obs = self.next_obs[: self.size]
        self.not_done = self.not_done[: self.size]

    def convert_minari(self, dataset: minari.MinariDataset):
        assert dataset.observation_space.shape[0] == self.obs.shape[-1], "Observation dimension mismatch."
        assert dataset.action_space.shape[0] == self.action.shape[-1], "Action dimension mismatch."

        for episode in dataset.iterate_episodes():
            transition = {
                "obs": episode.observations[:-1],
                "action": episode.actions,
                "next_obs": episode.observations[1:],
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
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
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
            self.set_dataset_statistics(np.array(stats["obs_mean"]), np.array(stats["obs_std"]))
        else:
            logging.warning(f"Dataset statistics not found at {stats_path}. Replay buffer will not be normalized.")

        return obs_mean, obs_std

    def compute_dataset_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and standard deviation of the observation dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - obs_mean (np.ndarray): The mean of the observations.
                - obs_std (np.ndarray): The standard deviation of the observations.
        """
        obs_mean = np.mean(self.obs[: self.size], axis=0, keepdims=False)
        obs_std = np.std(self.obs[: self.size], axis=0, keepdims=False)

        return obs_mean, obs_std

    def set_dataset_statistics(self, obs_mean: np.ndarray, obs_std: np.ndarray):
        """
        Set the mean and standard deviation for the dataset and normalize the observations.

        Args:
            obs_mean (np.ndarray): The mean to use for normalization.
            obs_std (np.ndarray): The standard deviation to use for normalization.
        """
        self.obs_mean = obs_mean
        self.obs_std = obs_std

        self.obs[: self.size] = normalize(self.obs[: self.size], self.obs_mean, self.obs_std)
        self.next_obs[: self.size] = normalize(self.next_obs[: self.size], self.obs_mean, self.obs_std)

    def get_dataset_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current dataset statistics (mean and standard deviation).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - obs_mean (np.ndarray): The mean of the observations.
                - obs_std (np.ndarray): The standard deviation of the observations.
        """
        return self.obs_mean, self.obs_std


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

    print("Mean and std shapes:", mean.shape, std.shape)

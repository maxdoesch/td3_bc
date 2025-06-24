import os
import numpy as np
import torch
import json
import minari
import logging
from typing import Dict, Tuple, Optional


def normalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-5):
    return (array - mean) / (std + eps)


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, max_size: int = int(1e7), device: Optional[str] = None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.obs = np.zeros((max_size, obs_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.obs_mean = np.zeros(obs_dim)
        self.obs_std = np.ones(obs_dim)

    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, done: np.ndarray):
        """
        Add transitions to the replay buffer in a vectorized way.

        Args:
            obs (np.ndarray): Unnormalized current obs, shape (n_env, obs_dim) or (obs_dim,)
            action (np.ndarray): Action taken, shape (n_env, action_dim) or (action_dim,)
            next_obs (np.ndarray): Unnormalize next obs, shape (n_env, obs_dim) or (obs_dim,)
            reward (np.ndarray): Reward received, shape (n_env, 1) or (1,)
            done (np.ndarray): Done flag, shape (n_env, 1) or (1,)
        """
        # Convert to batch if single transition
        obs = np.expand_dims(obs, axis=0) if obs.ndim == 1 else obs
        action = np.expand_dims(action, axis=0) if action.ndim == 1 else action
        next_obs = np.expand_dims(next_obs, axis=0) if next_obs.ndim == 1 else next_obs
        reward = np.expand_dims(reward, axis=1) if reward.ndim == 1 else reward
        done = np.expand_dims(done, axis=1) if done.ndim == 1 else done

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
                - "obs": Tensor of shape (batch_size, obs_dim) with normalized observations.
                - "action": Tensor of shape (batch_size, action_dim) with actions taken.
                - "next_obs": Tensor of shape (batch_size, obs_dim) with normalized next observations.
                - "reward": Tensor of shape (batch_size) with rewards received.
                - "not_done": Tensor of shape (batch_size) indicating whether the episode has not ended.
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.FloatTensor(self.obs[idx]).to(self.device),
            "action": torch.FloatTensor(self.action[idx]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(self.device),
            "reward": torch.FloatTensor(self.reward[idx]).to(self.device),
            "not_done": torch.FloatTensor(self.not_done[idx]).to(self.device),
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
        obs_mean = np.mean(self.obs[: self.size], axis=0, keepdims=True)
        obs_std = np.std(self.obs[: self.size], axis=0, keepdims=True)

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
    obs_dim = 3
    action_dim = 4
    max_size = int(1e6)
    n_env = 5

    buffer = ReplayBuffer(obs_dim, action_dim, max_size)

    # Simulate adding transitions
    transition = {
        "obs": np.random.rand(n_env, obs_dim),
        "action": np.random.rand(n_env, action_dim),
        "next_obs": np.random.rand(n_env, obs_dim),
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
        "obs": [np.random.randn(episode_length, obs_dim) for _ in range(episodes)],
        "next_obs": [np.random.randn(episode_length, obs_dim) for _ in range(episodes)],
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

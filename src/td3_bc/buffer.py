from typing import Dict, Tuple

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e7)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: np.ndarray, done: np.ndarray):
        """
        Add transitions to the replay buffer in a vectorized way.

        Args:
            state (np.ndarray): Current state, shape (n_env, state_dim) or (state_dim,)
            action (np.ndarray): Action taken, shape (n_env, action_dim) or (action_dim,)
            next_state (np.ndarray): Next state, shape (n_env, state_dim) or (state_dim,)
            reward (np.ndarray): Reward received, shape (n_env, 1) or (1,)
            done (np.ndarray): Done flag, shape (n_env, 1) or (1,)
        """
        # Convert to batch if single transition
        state = np.expand_dims(state, axis=0) if state.ndim == 1 else state
        action = np.expand_dims(action, axis=0) if action.ndim == 1 else action
        next_state = np.expand_dims(next_state, axis=0) if next_state.ndim == 1 else next_state
        reward = np.expand_dims(reward, axis=1) if reward.ndim == 1 else reward
        done = np.expand_dims(done, axis=1) if done.ndim == 1 else done

        assert state.shape[0] == action.shape[0] == next_state.shape[0] == reward.shape[0] == done.shape[0], (
            "All inputs must have the same first dimension (number of environments)."
        )

        n_env = state.shape[0]

        indices = np.arange(self.ptr, self.ptr + n_env) % self.max_size

        self.state[indices] = state
        self.action[indices] = action
        self.next_state[indices] = next_state
        self.reward[indices] = reward
        self.not_done[indices] = 1 - done

        self.ptr = (self.ptr + n_env) % self.max_size
        self.size = min(self.size + n_env, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "state": torch.FloatTensor(self.state[idx]).to(device),
            "action": torch.FloatTensor(self.action[idx]).to(device),
            "next_state": torch.FloatTensor(self.next_state[idx]).to(device),
            "reward": torch.FloatTensor(self.reward[idx]).to(device),
            "not_done": torch.FloatTensor(self.not_done[idx]).to(device),
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
            transition = {
                "state": dict_dataset["obs"][episode][:-1],
                "action": dict_dataset["acts"][episode],
                "next_state": dict_dataset["obs"][episode][1:],
                "reward": dict_dataset["rews"][episode],
                "done": np.concatenate(
                    [
                        np.zeros_like(dict_dataset["rews"][episode][:-1]),
                        np.ones_like(dict_dataset["rews"][episode][-1:]),
                    ]
                ),
            }

            self.add(**transition)

        self.state = self.state[: self.size]
        self.action = self.action[: self.size]
        self.reward = self.reward[: self.size]
        self.next_state = self.next_state[: self.size]
        self.not_done = self.not_done[: self.size]
        self.size = self.state.shape[0]

    def compute_dataset_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(self.state, axis=0, keepdims=True)
        std = np.std(self.state, axis=0, keepdims=True)

        return mean, std

    def normalize_states(self, mean: np.ndarray, std: np.ndarray, eps=1e-3):
        self.state = (self.state - mean) / (std + eps)
        self.next_state = (self.next_state - mean) / (std + eps)


if __name__ == "__main__":
    # Example usage
    state_dim = 3
    action_dim = 4
    max_size = int(1e6)
    n_env = 5

    buffer = ReplayBuffer(state_dim, action_dim, max_size)

    # Simulate adding transitions
    transition = {
        "state": np.random.rand(n_env, state_dim),
        "action": np.random.rand(n_env, action_dim),
        "next_state": np.random.rand(n_env, state_dim),
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
        "obs": [np.random.randn(episode_length + 1, state_dim) for _ in range(episodes)],
        "acts": [np.random.randn(episode_length, action_dim) for _ in range(episodes)],
        "rews": [np.random.randn(episode_length) for _ in range(episodes)],
    }
    buffer.convert_dict(dict_dataset)
    print("Buffer size after converting dictionary dataset:", buffer.size)
    print("Buffer pointer after converting dictionary dataset:", buffer.ptr)

    # Normalize states
    mean, std = buffer.normalize_states()
    print("Mean:", mean)
    print("Std:", std)

import os
import json
import math
import logging
from typing import Dict, Tuple, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
import h5py
import minari

import td3_bc.utils as utils

import threading


def normalize(array: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-3):
    return (array - mean) / (std + eps)


class ReplayBuffer:
    """
    Disk-backed replay buffer using HDF5. Keeps only small staging tensors in RAM.
    """

    def __init__(
        self,
        obs_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        max_size: int = int(1e6),
        device: Optional[str] = None,
        augmentations: bool = True,
        h5_path: str = "replay.h5",
        mode: str = "a",
        compression: Optional[str] = None,  # e.g. "lzf" or "gzip"
        chunk_mb: int = 8,
        row_chunks: int = 1, 
        rdcc_nbytes: int = 256*1024*1024, 
        rdcc_nslots: int = 1000003
    ):
        """
        Args:
            obs_shape: int or tuple; same semantics as your in-memory buffer.
            action_dim: number of action dims.
            max_size: capacity of the buffer (number of transitions).
            device: torch device for sampling outputs.
            augmentations: if True and obs are images, apply simple RandomCrop on sample().
            h5_path: file path for the HDF5 store.
            mode: h5py open mode; "a" creates if missing and reuses if present.
            compression: None, "lzf", or "gzip". Use None for max speed.
            chunk_mb: per-dataset chunk size target in megabytes (used to compute chunk shapes).
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Normalize obs_shape format
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)
        self.action_dim = int(action_dim)

        # Heuristic for image observations (channel or height/width present)
        self.is_image_obs = len(self.obs_shape) >= 2

        # Handle frame stacking provided in obs_shape like (F, C, H, W) or (F, H, W, C)
        self.frame_stack = 1
        if self.is_image_obs and len(self.obs_shape) >= 4:
            self.frame_stack = self.obs_shape[0]
            # Keep full shape as provided so we don't silently transpose user data.
            # Augmentations later will just crop on the last two dims.
        self.max_size = int(max_size) if max_size is not None else int(1e6)
        self.row_chunks = int(row_chunks)
        # Stats tensors (kept small on device)
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

        self._staging = None  # for pinned H2D transfers
        self.augmentations = (
            T.Compose([T.RandomCrop(self.obs_shape[-2:], padding=4, padding_mode="constant")])
            if (self.is_image_obs and augmentations)
            else None
        )
        self._io_lock = threading.RLock()
        # Open or initialize HDF5 file
        self.h5_path = h5_path
        # make chunks one row
        self.f =  h5py.File(self.h5_path, mode, rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
        self._init_or_validate_file(compression, chunk_mb)

        # Load pointer and size from file metadata if present
        self.ptr = int(self.f.attrs.get("ptr", 0))
        self.size = int(self.f.attrs.get("size", 0))

        

    # ---- HDF5 helpers ----

    def _compute_chunk_shape(self, elem_shape, dtype, chunk_mb: int):
        """Compute a chunk shape roughly chunk_mb megabytes in size, aligned on the first axis."""
        bytes_per = np.dtype(dtype).itemsize * int(np.prod(elem_shape))
        items_per_chunk = max(1, (chunk_mb * 1024 * 1024) // bytes_per)
        return (min(self.max_size, items_per_chunk), *elem_shape)

    def _require_dataset(self, name, shape, dtype, compression, chunk_mb):
        if name in self.f:
            ds = self.f[name]
            # Validate shape and dtype
            if ds.shape != (self.max_size, *shape) or ds.dtype != np.dtype(dtype):
                raise ValueError(
                    f"Existing dataset {name} has incompatible shape/dtype. "
                    f"Found {ds.shape}, {ds.dtype}, expected {(self.max_size, *shape)}, {dtype}"
                )
            return ds
        # Create new
        #chunks = self._compute_chunk_shape(shape, dtype, chunk_mb)
        chunks = (self.row_chunks, *shape)
        return self.f.create_dataset(
            name,
            shape=(self.max_size, *shape),
            maxshape=(self.max_size, *shape),
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )

    def _init_or_validate_file(self, compression, chunk_mb):

        if "obs" in self.f and isinstance(self.f["obs"], h5py.Dataset):
            file_max = int(self.f["obs"].shape[0])
            # adopt file capacity
            self.max_size = file_max
            # normalize attrs to file reality
            self.f.attrs["max_size"] = file_max
            # also validate shapes/dtypes for other datasets here and return
            self.d_obs  = self.f["obs"]
            self.d_next = self.f["next_obs"]
            self.d_act  = self.f["action"]
            self.d_rew  = self.f["reward"]
            self.d_nd   = self.f["not_done"]
            return
        
        # Store meta in file attrs for validation across runs
        self.f.attrs.setdefault("obs_shape", np.array(self.obs_shape, dtype=np.int64))
        self.f.attrs.setdefault("action_dim", int(self.action_dim))
        self.f.attrs.setdefault("max_size", int(self.max_size))

        # If file exists with existing meta, validate
        if tuple(self.f.attrs["obs_shape"]) != tuple(self.obs_shape):
            raise ValueError("HDF5 file obs_shape mismatch.")
        if int(self.f.attrs["action_dim"]) != self.action_dim:
            raise ValueError("HDF5 file action_dim mismatch.")
        if int(self.f.attrs["max_size"]) != self.max_size:
            raise ValueError("HDF5 file max_size mismatch.")

        # Dtypes
        obs_dtype = np.uint8 if self.is_image_obs else np.float32

        # Create/require datasets
        self.d_obs = self._require_dataset("obs", self.obs_shape, obs_dtype, compression, chunk_mb)
        self.d_next = self._require_dataset("next_obs", self.obs_shape, obs_dtype, compression, chunk_mb)
        self.d_act = self._require_dataset("action", (self.action_dim,), np.float32, compression, chunk_mb)
        self.d_rew = self._require_dataset("reward", (1,), np.float32, compression, chunk_mb)
        self.d_nd = self._require_dataset("not_done", (1,), np.float32, compression, chunk_mb)

    def _flush_meta(self):
        self.f.attrs["ptr"] = int(self.ptr)
        self.f.attrs["size"] = int(self.size)
        self.f.flush()

    def _read_fancy(self, dset, idx: np.ndarray) -> np.ndarray:
        # idx: shape (B,), dtype int64
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        # unique indices for the actual h5 read (avoids issues with duplicates)
        uniq_idx, inverse = np.unique(sorted_idx, return_inverse=True)
        arr_uniq = dset[uniq_idx]  # h5py read with increasing order indices
        arr_sorted = arr_uniq[inverse]  # re-expand duplicates, still sorted by idx
        # unsort to match original random order
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)
        return arr_sorted[inv_order]

    def _read_grouped(self, dset, idx: np.ndarray) -> np.ndarray:
        # idx: (B,) int64
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]

        # find contiguous runs in sorted_idx
        diffs = np.diff(sorted_idx)
        run_starts = np.concatenate(([0], np.nonzero(diffs != 1)[0] + 1))
        run_ends = np.concatenate((run_starts[1:], [sorted_idx.size]))

        # read each run as one contiguous slice
        out_sorted = np.empty((idx.shape[0],) + dset.shape[1:], dtype=dset.dtype)
        pos = 0
        for s, e in zip(run_starts, run_ends):
            start = int(sorted_idx[s])
            n = int(e - s)
            block = dset[start:start + n]    # contiguous slice read
            out_sorted[pos:pos + n] = block
            pos += n

        # unsort back to original random order
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return out_sorted[inv]
    # ---- Public API (mirrors your original) ----

    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, done: np.ndarray):
        """
        Vectorized add.
        Shapes follow your original: obs (n, *obs_shape) or (*obs_shape,), action (n, A) or (A,),
        reward (n, 1) or (1,), done same.
        """
        obs = np.expand_dims(obs, 0) if obs.ndim == len(self.obs_shape) else obs
        next_obs = np.expand_dims(next_obs, 0) if next_obs.ndim == len(self.obs_shape) else next_obs
        action = np.expand_dims(action, 0) if action.ndim == 1 else action
        reward = np.expand_dims(reward, 1) if reward.ndim == 1 else reward
        done = np.expand_dims(done, 1) if done.ndim == 1 else done

        n_env = obs.shape[0]
        assert (
            action.shape[0] == next_obs.shape[0] == reward.shape[0] == done.shape[0] == n_env
        ), "All inputs must share the same batch dimension."

        with self._io_lock:
            # Compute write indices with wrap-around
            start = self.ptr
            end = self.ptr + n_env
            if end <= self.max_size:
                sl = slice(start, end)
                self._assign_slice(sl, obs, next_obs, action, reward, done)
            else:
                # Wrap around
                first = slice(start, self.max_size)
                second = slice(0, end % self.max_size)
                split = self.max_size - start
                self._assign_slice(first, obs[:split], next_obs[:split], action[:split], reward[:split], done[:split])
                self._assign_slice(second, obs[split:], next_obs[split:], action[split:], reward[split:], done[split:])

            self.ptr = (self.ptr + n_env) % self.max_size
            self.size = min(self.size + n_env, self.max_size)
            self._flush_meta()

    def _assign_slice(self, sl: slice, obs, next_obs, action, reward, done):
        if self.is_image_obs:
            obs = obs.astype(np.uint8, copy=False)
            next_obs = next_obs.astype(np.uint8, copy=False)
        else:
            obs = obs.astype(np.float32, copy=False)
            next_obs = next_obs.astype(np.float32, copy=False)

        self.d_obs[sl] = obs
        self.d_next[sl] = next_obs
        self.d_act[sl] = action.astype(np.float32, copy=False)
        self.d_rew[sl] = reward.astype(np.float32, copy=False)
        self.d_nd[sl] = (1.0 - done.astype(np.float32, copy=False))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Random sample from disk. Uses pinned RAM staging then async H2D copy.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample: buffer is empty.")

        idx = np.random.randint(0, self.size, size=(batch_size,), dtype=np.int64)
        import time
        t0 = time.time()
        with self._io_lock:
            # Fancy-index read from HDF5 (returns numpy arrays)
            obs_np  = self._read_grouped(self.d_obs,  idx)
            next_np = self._read_grouped(self.d_next, idx)
            act_np  = self._read_grouped(self.d_act,  idx)
            rew_np  = self._read_grouped(self.d_rew,  idx)
            nd_np   = self._read_grouped(self.d_nd,   idx)

        t_read = time.time() - t0
        t1 = time.time()
        # Create (or resize) pinned staging tensors
        if self._staging is None or self._staging["obs"].shape[0] != batch_size:
            obs_dtype = torch.uint8 if self.is_image_obs else torch.float32
            self._staging = {
                "obs": torch.empty((batch_size, *self.obs_shape), dtype=obs_dtype, pin_memory=True),
                "next_obs": torch.empty((batch_size, *self.obs_shape), dtype=obs_dtype, pin_memory=True),
                "action": torch.empty((batch_size, self.action_dim), dtype=torch.float32, pin_memory=True),
                "reward": torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True),
                "not_done": torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True),
            }

        # Copy numpy -> pinned tensors
        self._staging["obs"].copy_(torch.from_numpy(obs_np), non_blocking=False)
        self._staging["next_obs"].copy_(torch.from_numpy(next_np), non_blocking=False)
        self._staging["action"].copy_(torch.from_numpy(act_np), non_blocking=False)
        self._staging["reward"].copy_(torch.from_numpy(rew_np), non_blocking=False)
        self._staging["not_done"].copy_(torch.from_numpy(nd_np), non_blocking=False)

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

        t_move = time.time() - t1

        print(f"[SAMPLE] batch={batch_size} read={t_read:.3f}s move={t_move:.3f}s")

        return {
            "obs": obs_norm,
            "action": action,
            "next_obs": next_obs_norm,
            "reward": reward,
            "not_done": not_done,
        }

    def _get_stacked_observations(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Same logic as your original
        if self.frame_stack > 1:
            obs_padded = np.concatenate([np.repeat(obs[:1], self.frame_stack - 1, axis=0), obs], axis=0)
            obs_sw = np.lib.stride_tricks.sliding_window_view(obs_padded[:-1], window_shape=self.frame_stack, axis=0)
            obs_sw = np.moveaxis(obs_sw, -1, 1)
            next_sw = np.lib.stride_tricks.sliding_window_view(obs_padded[1:], window_shape=self.frame_stack, axis=0)
            next_sw = np.moveaxis(next_sw, -1, 1)
            return obs_sw, next_sw
        else:
            return obs[:-1], obs[1:]

    def convert_dict(self, dict_dataset):
        """
        Stream episodes from a dictionary dataset onto disk.
        dict_dataset keys:
            "obs": list of arrays [T+1, *obs_shape]
            "acts": list of arrays [T, A]
            "rews": list of arrays [T]
        """
        n_eps = len(dict_dataset["acts"])
        for ep in range(n_eps):
            obs = np.array(dict_dataset["obs"][ep])
            acts = np.array(dict_dataset["acts"][ep])
            rews = np.array(dict_dataset["rews"][ep])
            done = np.concatenate([np.zeros_like(rews[:-1]), np.ones_like(rews[-1:])])

            obs, next_obs = self._get_stacked_observations(obs)
            transition = {
                "obs": obs,
                "action": acts,
                "next_obs": next_obs,
                "reward": rews.reshape(-1, 1),
                "done": done.reshape(-1, 1),
            }
            self.add(**transition)

    def convert_minari(self, dataset: minari.MinariDataset):
        assert dataset.action_space.shape[0] == self.action_dim, "Action dimension mismatch."
        for episode in dataset.iterate_episodes():
            observations = utils.uncombine_stacked_frames(episode.observations)
            obs, next_obs = self._get_stacked_observations(observations)
            transition = {
                "obs": obs,
                "action": episode.actions,
                "next_obs": next_obs,
                "reward": episode.rewards.reshape(-1, 1),
                "done": episode.terminations.reshape(-1, 1),
            }
            self.add(**transition)

    # ---- Statistics (computed streaming to avoid RAM spikes) ----

    def compute_dataset_statistics(self, batch: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std over stored observations on disk.
        Uses Welford’s algorithm with chunked reads.
        """
        if self.is_image_obs:
            obs_mean = np.array(0.0, dtype=np.float32)
            obs_std = np.array(255.0, dtype=np.float32)
            return obs_mean, obs_std

        count = 0
        mean = None
        M2 = None

        total = self.size
        if total == 0:
            # Fallback to zeros/ones like init
            return self.obs_mean.detach().cpu().numpy(), self.obs_std.detach().cpu().numpy()

        # Iterate in chunks over the valid segment [0, size)
        for start in range(0, total, batch):
            end = min(start + batch, total)
            x = self.d_obs[start:end].astype(np.float32, copy=False)  # shape [B, *obs_shape]

            # Flatten batch dimension only
            b = x.shape[0]
            x = x.reshape(b, *self.obs_shape)

            # Batch statistics
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = b

            if mean is None:
                mean = batch_mean
                M2 = batch_var * batch_count
                count = batch_count
            else:
                delta = batch_mean - mean
                tot = count + batch_count
                mean = mean + delta * (batch_count / tot)
                M2 = M2 + batch_var * batch_count + delta * delta * (count * batch_count / tot)
                count = tot

        var = M2 / max(1, count - 1)
        std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)

        return mean.astype(np.float32), std

    def set_dataset_statistics(self, obs_mean: np.ndarray, obs_std: np.ndarray):
        self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.obs_std = torch.tensor(obs_std, dtype=torch.float32, device=self.device)

    def get_dataset_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.obs_mean.detach().cpu().numpy(), self.obs_std.detach().cpu().numpy()

    def save_statistics(self, stats_path: str):
        if not stats_path.endswith(".json"):
            stats_path = os.path.join(stats_path, "dataset_statistics.json")
        stats = {"obs_mean": self.get_dataset_statistics()[0].tolist(),
                 "obs_std": self.get_dataset_statistics()[1].tolist()}
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

    def load_statistics(self, stats_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not stats_path.endswith(".json"):
            stats_path = os.path.join(stats_path, "dataset_statistics.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            obs_mean = np.array(stats["obs_mean"], dtype=np.float32)
            obs_std = np.array(stats["obs_std"], dtype=np.float32)
            self.set_dataset_statistics(obs_mean, obs_std)
        else:
            logging.warning(f"Dataset statistics not found at {stats_path}. Replay buffer will not be normalized.")
            obs_mean = self.obs_mean.detach().cpu().numpy()
            obs_std = self.obs_std.detach().cpu().numpy()
        return obs_mean, obs_std

    # ---- Cleanup ----

    def close(self):
        self._flush_meta()
        try:
            self.f.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Minimal smoke test
    obs_shape = 3
    action_dim = 4
    max_size = int(1e5)
    buf = ReplayBuffer(obs_shape, action_dim, max_size=max_size, h5_path="rb_small.h5", compression=None)

    # Add a small batch
    n_env = 7
    transition = {
        "obs": np.random.rand(n_env, obs_shape).astype(np.float32),
        "action": np.random.rand(n_env, action_dim).astype(np.float32),
        "next_obs": np.random.rand(n_env, obs_shape).astype(np.float32),
        "reward": np.random.rand(n_env, 1).astype(np.float32),
        "done": np.random.randint(0, 2, size=(n_env, 1)).astype(np.float32),
    }
    buf.add(**transition)
    print("size:", buf.size, "ptr:", buf.ptr)

    # Stats
    mean, std = buf.compute_dataset_statistics()
    buf.set_dataset_statistics(mean, std)
    print("mean shape:", np.shape(mean), "std shape:", np.shape(std))

    # Sample
    batch = buf.sample(batch_size=4)
    for k, v in batch.items():
        print(k, tuple(v.shape))

    buf.close()

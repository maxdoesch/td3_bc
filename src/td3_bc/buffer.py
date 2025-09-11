import os
import json
import logging
from typing import Dict, Tuple, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
import h5py
import minari
import threading

import td3_bc.utils as utils


def normalize(array: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-3):
    return (array - mean) / (std + eps)


class ReplayBuffer:
    """
    Disk-backed replay buffer using a single HDF5 dataset ("transitions") with a compound dtype.
    Each row contains: obs, next_obs, action, reward, not_done.
    Sampling uses one fancy-index read (on sorted unique indices), then reorders in RAM.
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
        row_chunks: int = 64,               # rows per chunk; tune for your storage
        rdcc_nbytes: int = 512 * 1024 * 1024,  # HDF5 raw data chunk cache
        rdcc_nslots: int = 2_000_003,
        swmr: bool = False,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Normalize shapes
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)
        self.action_dim = int(action_dim)

        # Image heuristic
        self.is_image_obs = len(self.obs_shape) >= 2

        # Frame stacking
        self.frame_stack = 1
        if self.is_image_obs and len(self.obs_shape) >= 4:
            self.frame_stack = self.obs_shape[0]

        self.max_size = int(max_size)
        self.row_chunks = int(row_chunks)

        # Stats tensors (small, on device)
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

        self._staging = None  # pinned staging for H2D
        self.augmentations = (
            T.Compose([T.RandomCrop(self.obs_shape[-2:], padding=4, padding_mode="constant")])
            if (self.is_image_obs and augmentations)
            else None
        )

        self.h5_path = h5_path
        self._io_lock = threading.RLock()

        # open file
        self.f = h5py.File(self.h5_path, mode, rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, libver="latest", swmr=(swmr if mode == "r" else False))
        self._init_or_validate_file(compression)

        # load ptr/size
        self.ptr = int(self.f.attrs.get("ptr", 0))
        self.size = int(self.f.attrs.get("size", 0))

    # ---------- HDF5 init (single dataset) ----------

    def _make_row_dtype(self):
        obs_dtype = np.uint8 if self.is_image_obs else np.float32
        dt = np.dtype([
            ("obs",      obs_dtype,        self.obs_shape),
            ("next_obs", obs_dtype,        self.obs_shape),
            ("action",   np.float32,      (self.action_dim,)),
            ("reward",   np.float32,      (1,)),
            ("not_done", np.float32,      (1,)),
        ])
        return dt

    def _init_or_validate_file(self, compression):
        # If dataset already exists, adopt capacity and bind handle
        if "transitions" in self.f and isinstance(self.f["transitions"], h5py.Dataset):
            dset = self.f["transitions"]
            self.max_size = int(dset.shape[0])
            # only write attrs if file is writable
            if self._can_write():
                self.f.attrs["max_size"] = self.max_size
            self.dset = dset
            return

        # From here on we expect a writable, brand-new file
        if not self._can_write():
            raise RuntimeError(
                "HDF5 file is read-only but 'transitions' dataset is missing; "
                "create/convert the file in write mode first."
            )

        # store meta once
        self.f.attrs.setdefault("obs_shape", np.array(self.obs_shape, dtype=np.int64))
        self.f.attrs.setdefault("action_dim", int(self.action_dim))
        self.f.attrs.setdefault("max_size", int(self.max_size))

        # create compound dataset with row chunks
        dtype = self._compound_dtype()
        self.dset = self.f.create_dataset(
            "transitions",
            shape=(self.max_size,),
            maxshape=(self.max_size,),
            dtype=dtype,
            chunks=(self.row_chunks,),        # <- rows per chunk; you already have self.row_chunks
            compression=compression,
        )

    # --- add this helper in ReplayBuffer ---
    def _can_write(self) -> bool:
        # h5py modes that allow writes
        return getattr(self.f, "mode", "r") in ("r+", "a", "w", "w-")


    def _flush_meta(self):
        if not self._can_write():
            return
        self.f.attrs["ptr"] = int(self.ptr)
        self.f.attrs["size"] = int(self.size)
        self.f.flush()



    # ---------- Writes ----------

    def _assign_slice(self, sl: slice, obs, next_obs, action, reward, done):
        # dtype conversions
        if self.is_image_obs:
            obs = obs.astype(np.uint8, copy=False)
            next_obs = next_obs.astype(np.uint8, copy=False)
        else:
            obs = obs.astype(np.float32, copy=False)
            next_obs = next_obs.astype(np.float32, copy=False)

        action = action.astype(np.float32, copy=False)
        reward = reward.astype(np.float32, copy=False)
        not_done = (1.0 - done.astype(np.float32, copy=False))

        n = len(range(*sl.indices(self.max_size)))
        rows = np.empty(n, dtype=self.dset.dtype)
        rows["obs"] = obs
        rows["next_obs"] = next_obs
        rows["action"] = action
        rows["reward"] = reward
        rows["not_done"] = not_done
        self.dset[sl] = rows

    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, done: np.ndarray):
        """
        Vectorized add, same shapes as before:
          obs (n, *obs_shape) or (*obs_shape,)
          action (n, A) or (A,)
          reward (n, 1) or (1,)
          done (n, 1) or (1,)
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
            start = self.ptr
            end = self.ptr + n_env
            if end <= self.max_size:
                self._assign_slice(slice(start, end), obs, next_obs, action, reward, done)
            else:
                first = slice(start, self.max_size)
                second = slice(0, end % self.max_size)
                split = self.max_size - start
                self._assign_slice(first,  obs[:split], next_obs[:split], action[:split], reward[:split], done[:split])
                self._assign_slice(second, obs[split:], next_obs[split:], action[split:], reward[split:], done[split:])

            self.ptr = (self.ptr + n_env) % self.max_size
            self.size = min(self.size + n_env, self.max_size)
            self._flush_meta()

    # ---------- Reads ----------

    def read_block_np(self, start: int, count: int) -> np.ndarray:
        """Read a contiguous block of rows [start:start+count] as a compound numpy array."""
        end = min(start + count, self.size)
        if end <= start:
            raise RuntimeError("read_block_np requested empty range.")
        with self._io_lock:
            return self.dset[start:end]

    def read_indices_np(self, idx: np.ndarray) -> np.ndarray:
        """Read arbitrary rows in a single grouped pass (one dataset), returns compound numpy array."""
        with self._io_lock:
            return self._read_grouped_rows(self.dset, idx.astype(np.int64, copy=False))

    def _read_rows_one_call(self, dset, idx: np.ndarray):
        """
        ONE h5py call using sorted unique indices,
        then expand duplicates and restore original order.
        """
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        uniq_idx, inverse = np.unique(sorted_idx, return_inverse=True)

        # single fancy read (must be strictly increasing)
        rows_uniq = dset[uniq_idx]
        rows_sorted = rows_uniq[inverse]  # restore duplicates (still sorted)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)
        return rows_sorted[inv_order]     # back to original random order

    def _read_grouped_rows(self, dset, idx: np.ndarray):
        """Read rows using a small number of contiguous slices, then restore original order."""
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]

        # find contiguous runs
        diffs = np.diff(sorted_idx)
        run_starts = np.concatenate(([0], np.nonzero(diffs != 1)[0] + 1))
        run_ends   = np.concatenate((run_starts[1:], [sorted_idx.size]))

        out_sorted = np.empty(sorted_idx.shape[0], dtype=dset.dtype)
        pos = 0
        for s, e in zip(run_starts, run_ends):
            start = int(sorted_idx[s])
            n     = int(e - s)
            out_sorted[pos:pos+n] = dset[start:start+n]   # one contiguous slice read
            pos += n

        # unsort back to original random order
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return out_sorted[inv]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample: buffer is empty.")

        idx = np.random.randint(0, self.size, size=(batch_size,), dtype=np.int64)
        import time
        t0 = time.time()
        # Single dataset read
        with self._io_lock:
            rows = self._read_grouped_rows(self.dset, idx)
        t_read = time.time() - t0
        t1 = time.time()
        # Extract fields (numpy views)
        obs_np      = rows["obs"]
        next_np     = rows["next_obs"]
        action_np   = rows["action"]
        reward_np   = rows["reward"]
        not_done_np = rows["not_done"]

        # Allocate pinned staging (once per batch size)
        if self._staging is None or self._staging["obs"].shape[0] != batch_size:
            obs_dtype = torch.uint8 if self.is_image_obs else torch.float32
            self._staging = {
                "obs": torch.empty((batch_size, *self.obs_shape), dtype=obs_dtype, pin_memory=True),
                "next_obs": torch.empty((batch_size, *self.obs_shape), dtype=obs_dtype, pin_memory=True),
                "action": torch.empty((batch_size, self.action_dim), dtype=torch.float32, pin_memory=True),
                "reward": torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True),
                "not_done": torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True),
            }

        # numpy -> pinned
        self._staging["obs"].copy_(torch.from_numpy(obs_np), non_blocking=False)
        self._staging["next_obs"].copy_(torch.from_numpy(next_np), non_blocking=False)
        self._staging["action"].copy_(torch.from_numpy(action_np), non_blocking=False)
        self._staging["reward"].copy_(torch.from_numpy(reward_np), non_blocking=False)
        self._staging["not_done"].copy_(torch.from_numpy(not_done_np), non_blocking=False)

        # H2D
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
        t_norm = time.time() - t1

        print(f"[SAMPLE] batch={batch_size} read={t_read:.3f}s norm={t_norm:.3f}s")

        return {
            "obs": obs_norm,
            "action": action,
            "next_obs": next_obs_norm,
            "reward": reward,
            "not_done": not_done,
        }

    # ---------- Converters ----------

    def _get_stacked_observations(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    # ---------- Statistics ----------

    def compute_dataset_statistics(self, batch: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_image_obs:
            return np.array(0.0, dtype=np.float32), np.array(255.0, dtype=np.float32)

        total = self.size
        if total == 0:
            return self.obs_mean.detach().cpu().numpy(), self.obs_std.detach().cpu().numpy()

        count = 0
        mean = None
        M2 = None

        for start in range(0, total, batch):
            end = min(start + batch, total)
            x = self.dset[start:end]["obs"].astype(np.float32, copy=False)  # (B, *obs_shape)
            b = x.shape[0]
            x = x.reshape(b, *self.obs_shape)
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

    # ---------- Cleanup ----------

    def close(self):
        # only try to flush meta if file is writable
        if getattr(self, "f", None) is not None and self._can_write():
            try:
                self._flush_meta()
            except Exception:
                pass
        try:
            if getattr(self, "f", None) is not None:
                self.f.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

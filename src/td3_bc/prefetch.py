# td3_bc/prefetch.py

import threading
import queue
import time
from typing import Dict, Optional, List
import numpy as np
import torch


class H5Prefetcher:
    """
    Double-buffered prefetcher:
      - Loader thread reads a mega-batch (M rows) into pinned CPU tensors (the "back" buffer).
      - Pusher thread slices the "front" buffer into per-step batches and pushes them to a small queue.
      - While pusher is feeding the queue, loader prepares the next mega-batch in parallel.
    """

    def __init__(
        self,
        buffer,
        batch_size: int,
        device: str,
        # queue holds per-step batches; keep small to avoid huge GPU stalls
        max_prefetch: int = 8,
        # Either specify M directly (rows per mega-batch), or give a RAM budget:
        mega_batch_rows: Optional[int] = None,
        budget_bytes: Optional[int] = None,   # e.g. int(16e9) for ~16 GB total (front+back)
        # I/O mode
        read_contiguous: bool = True,         # contiguous reads are fastest
        profile: bool = False,
        # When the front buffer is nearly drained, start loading the back if not yet started:
        trigger_load_when_remaining_leq: int = 2,   # in *per-step batches*
    ):
        self.buffer = buffer
        self.batch_size = int(batch_size)
        self.device = device
        self.max_prefetch = int(max_prefetch)
        self.read_contiguous = bool(read_contiguous)
        self.profile = bool(profile)
        self.trigger_load_when_remaining_leq = int(trigger_load_when_remaining_leq)

        # Work queue of ready per-step batches for the learner thread
        self.q: "queue.Queue[Dict[str, torch.Tensor] | Exception]" = queue.Queue(maxsize=self.max_prefetch)

        # Double-buffer state
        self._front_mb = None     # dict of pinned CPU tensors (obs, next_obs, action, reward, not_done)
        self._front_M = 0
        self._front_off = 0       # row offset inside front mega-batch

        self._back_mb = None
        self._back_M = 0

        # Synchronization
        self._stop = threading.Event()
        self._back_ready = threading.Event()      # set when _back_mb filled
        self._load_requested = threading.Event()  # set when pusher wants loader to fill back
        self._loading = threading.Event()         # loader is currently reading

        # threads
        self._loader_t = None
        self._pusher_t = None

        # Determine mega-batch rows (M)
        self._row_bytes = int(self.buffer.dset.dtype.itemsize)  # compound row size in bytes
        self._M = self._choose_mega_rows(mega_batch_rows, budget_bytes)
        if self._M < self.batch_size:
            raise RuntimeError(
                f"Mega-batch size M={self._M} is smaller than batch_size={self.batch_size}. "
                f"Increase budget_bytes or reduce batch_size."
            )

    # ---------- sizing helpers ----------

    def _choose_mega_rows(self, mega_batch_rows, budget_bytes) -> int:
        """Pick M (rows per mega-batch) given either explicit rows or a RAM budget."""
        size = max(1, int(self.buffer.size))  # avoid zero in online case
        B = self.batch_size

        if mega_batch_rows is not None:
            M = int(mega_batch_rows)
        elif budget_bytes is not None:
            # Two buffers in RAM: ~ budget_bytes / 2 each
            per_buffer_budget = max(1, int(budget_bytes) // 2)
            rows_per_buffer = max(1, per_buffer_budget // self._row_bytes)
            M = rows_per_buffer
        else:
            # default: read 8 training batches per mega-batch
            M = 8 * B

        # Align to whole batches and not exceed dataset
        M = max(B, (M // B) * B)
        M = min(M, (size // B) * B) if size >= B else B
        return M

    # ---------- pinned buffer helpers ----------

    def _alloc_mb(self, M: int):
        obs_dtype = torch.uint8 if self.buffer.is_image_obs else torch.float32
        return {
            "obs": torch.empty((M, *self.buffer.obs_shape), dtype=obs_dtype, pin_memory=True),
            "next_obs": torch.empty((M, *self.buffer.obs_shape), dtype=obs_dtype, pin_memory=True),
            "action": torch.empty((M, self.buffer.action_dim), dtype=torch.float32, pin_memory=True),
            "reward": torch.empty((M, 1), dtype=torch.float32, pin_memory=True),
            "not_done": torch.empty((M, 1), dtype=torch.float32, pin_memory=True),
        }

    def _rows_to_mb(self, rows, mb):
        # numpy → pinned CPU
        mb["obs"].copy_(torch.from_numpy(rows["obs"]), non_blocking=False)
        mb["next_obs"].copy_(torch.from_numpy(rows["next_obs"]), non_blocking=False)
        mb["action"].copy_(torch.from_numpy(rows["action"]), non_blocking=False)
        mb["reward"].copy_(torch.from_numpy(rows["reward"]), non_blocking=False)
        mb["not_done"].copy_(torch.from_numpy(rows["not_done"]), non_blocking=False)

    # ---------- loader thread ----------

    def _load_back(self):
        try:
            while not self._stop.is_set():
                # wait until pusher requests a load
                if not self._load_requested.wait(timeout=0.01):
                    continue
                self._load_requested.clear()
                if self._stop.is_set():
                    break

                self._loading.set()

                # Decide rows to read
                size = int(self.buffer.size)
                if size < self.batch_size:
                    # not enough to form 1 batch (shouldn't happen offline)
                    self._loading.clear()
                    time.sleep(0.01)
                    continue

                # Adjust M if dataset small
                M = min(self._M, (size // self.batch_size) * self.batch_size)

                t0 = time.time()
                # Read from HDF5
                if self.read_contiguous and size >= M:
                    start = np.random.randint(0, size - M + 1)
                    rows = self.buffer.read_block_np(start, M)
                else:
                    idx = np.random.randint(0, size, size=M, dtype=np.int64)
                    rows = self.buffer.read_indices_np(idx)
                t_read = time.time() - t0

                # Convert into pinned back buffer
                t1 = time.time()
                back_mb = self._alloc_mb(M)
                self._rows_to_mb(rows, back_mb)
                t_pack = time.time() - t1

                self._back_mb = back_mb
                self._back_M = M
                self._back_ready.set()

                if self.profile:
                    print(f"[PREFETCH:LOAD] M={M} read={t_read:.3f}s pack={t_pack:.3f}s")
                self._loading.clear()
        except Exception as e:
            # Surface to consumer
            self.q.put(e)

    # ---------- pusher thread ----------

    def _push_front_batches(self):
        try:
            B = self.batch_size
            while not self._stop.is_set():
                # Ensure we have a front buffer
                if self._front_mb is None:
                    # If no back prepared yet, request and wait once
                    if not self._back_ready.is_set():
                        self._load_requested.set()
                        if not self._back_ready.wait(timeout=200.0):
                            raise TimeoutError("Timed out waiting for first mega-batch to load.")
                    # swap back → front
                    self._front_mb, self._front_M = self._back_mb, self._back_M
                    self._front_off = 0
                    self._back_mb = None
                    self._back_M = 0
                    self._back_ready.clear()

                # Start loading the back buffer early (overlap) when we're close to draining front
                remaining_batches = ((self._front_M - self._front_off) // B)
                if (remaining_batches <= self.trigger_load_when_remaining_leq) and \
                   (not self._back_ready.is_set()) and (not self._loading.is_set()):
                    self._load_requested.set()

                # If front has data, slice next B rows and push to queue
                if self._front_off < self._front_M:
                    s = slice(self._front_off, self._front_off + B)
                    self._front_off += B

                    # Move to device + normalize/augment
                    obs = self._front_mb["obs"][s].to(self.device, non_blocking=True)
                    nxt = self._front_mb["next_obs"][s].to(self.device, non_blocking=True)
                    if self.buffer.is_image_obs:
                        obs = obs.float()
                        nxt = nxt.float()

                    act = self._front_mb["action"][s].to(self.device, non_blocking=True)
                    rew = self._front_mb["reward"][s].to(self.device, non_blocking=True)
                    nd  = self._front_mb["not_done"][s].to(self.device, non_blocking=True)

                    if self.buffer.is_image_obs and self.buffer.augmentations:
                        obs = self.buffer.augmentations(obs)
                        nxt = self.buffer.augmentations(nxt)

                    obs = (obs - self.buffer.obs_mean) / (self.buffer.obs_std + 1e-3)
                    nxt = (nxt - self.buffer.obs_mean) / (self.buffer.obs_std + 1e-3)

                    batch = {"obs": obs, "action": act, "next_obs": nxt, "reward": rew, "not_done": nd}
                    # block if queue full; loader can still run in parallel
                    self.q.put(batch)
                else:
                    # Front drained; if back ready, swap and continue; else request load and wait briefly
                    if self._back_ready.is_set():
                        self._front_mb, self._front_M = self._back_mb, self._back_M
                        self._front_off = 0
                        self._back_mb = None
                        self._back_M = 0
                        self._back_ready.clear()
                    else:
                        self._load_requested.set()
                        # Small wait to avoid busy loop; the queue likely still has items keeping trainer busy
                        time.sleep(0.005)
        except Exception as e:
            self.q.put(e)

    # ---------- public API ----------

    def start(self):
        if self._loader_t and self._loader_t.is_alive():
            return
        self._stop.clear()
        self._back_ready.clear()
        self._load_requested.clear()
        self._loading.clear()
        self._loader_t = threading.Thread(target=self._load_back, daemon=True)
        self._pusher_t = threading.Thread(target=self._push_front_batches, daemon=True)
        self._loader_t.start()
        self._pusher_t.start()

    def get(self, timeout=None) -> Dict[str, torch.Tensor]:
        item = self.q.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        self._stop.set()
        if self._loader_t:
            self._loader_t.join(timeout=5)
            self._loader_t = None
        if self._pusher_t:
            self._pusher_t.join(timeout=5)
            self._pusher_t = None
        # drain queue to avoid stale batches on restart
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        # release references to pinned buffers
        self._front_mb = None
        self._back_mb = None

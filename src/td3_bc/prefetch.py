import threading, queue
from typing import Dict
import torch

class H5Prefetcher:
    def __init__(self, buffer, batch_size: int, device: str, max_prefetch: int = 4):
        self.buffer = buffer
        self.batch_size = batch_size
        self.device = device
        self.q: "queue.Queue[Dict[str, torch.Tensor]]" = queue.Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _worker(self):
        try:
            while not self.stop_event.is_set():
                batch = self.buffer.sample(self.batch_size)
                # Use timeout so stop() can't deadlock when queue is full
                while not self.stop_event.is_set():
                    try:
                        self.q.put(batch, timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            # Surface error to consumer
            try:
                self.q.put(e, timeout=0.1)
            except Exception:
                pass

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()                 # <<< crucial for restart
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def get(self, timeout=None) -> Dict[str, torch.Tensor]:
        item = self.q.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        self.stop_event.set()
        # Drain queue to unblock a producer stuck on put()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None

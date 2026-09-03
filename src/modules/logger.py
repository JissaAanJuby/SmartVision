import csv
import queue
import threading
from datetime import datetime
from pathlib import Path


class FatigueLogger:
    """Background CSV writer for fatigue state transitions.

    log_event() only enqueues a row (a queue.put is effectively O(1) and
    never touches disk), so the video/detection loop is never blocked on
    file I/O. A single daemon thread drains the queue and appends rows.
    """

    def __init__(self, log_path, queue_maxsize: int = 1000):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: "queue.Queue[list]" = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._ensure_header()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _ensure_header(self):
        needs_header = (not self.log_path.exists()) or self.log_path.stat().st_size == 0
        if needs_header:
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "state", "ear", "mar", "pitch", "yaw"]
                )

    def log_event(self, state: str, ear: float, mar: float, pitch: float, yaw: float):
        """Non-blocking. Safe to call from the detection thread on every state transition."""
        row = [
            datetime.now().isoformat(timespec="seconds"),
            state,
            round(float(ear), 3),
            round(float(mar), 3),
            round(float(pitch), 1),
            round(float(yaw), 1),
        ]
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Never let logging backpressure the detection loop; drop oldest and retry once.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(row)
            except queue.Empty:
                pass

    def _worker(self):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                writer.writerow(row)
                f.flush()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)
import time
from abc import ABC
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict


class BaseWriter(ABC):
    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name

        self._buffers: Dict[str, deque] = {}
        self._buffers_max_seconds: Dict[str, int] = {}  # in seconds
        self._buffers_timestamps = defaultdict(list)
        self._buffers_approx_fps: Dict[str, float] = {}
        self._file = None

    def create_file_for_buffer(self, subfolder: str, bufname: str):
        raise NotImplementedError()

    def init_buffer(self, name: str, max_seconds: int):
        if max_seconds > 0:
            self._buffers[name] = deque()
            self._buffers_max_seconds[name] = max_seconds

    def add_to_buffer(self, name: str, frame):
        if self._buffers[name] is None:
            return

        timestamp = time.time()
        self._buffers_timestamps[name].append(timestamp)

        # Calculate time window based on max_seconds
        time_window = self._buffers_max_seconds[name]

        # Remove frames that fall outside the time window
        while self._buffers_timestamps[name] and (timestamp - self._buffers_timestamps[name][0] > time_window):
            self._buffers[name].popleft()
            self._buffers_timestamps[name].pop(0)

        self._buffers[name].append(frame)

    def is_buffer_full(self, name: str) -> bool:
        if self._buffers[name].maxlen:
            return len(self._buffers[name]) == self._buffers[name].maxlen

        if not self._buffers_timestamps[name]:
            return False

        diff = self._buffers_timestamps[name][0] + self._buffers_max_seconds[name] - self._buffers_timestamps[name][-1]
        return diff < 0.1

    def is_buffer_empty(self, name: str) -> bool:
        return len(self._buffers[name]) == 0

    def write_from_buffer(self, name: str, n_elems: int):
        while len(self._buffers[name]) > 0 and n_elems > 0:
            frame = self._buffers[name].popleft()
            self.write(frame)
            n_elems -= 1

    def write(self, frame):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

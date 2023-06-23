from abc import ABC
from collections import deque
from pathlib import Path
from typing import Dict


class BaseWriter(ABC):
    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name

        self._buffers: Dict[str, deque] = {}
        self._file = None
        self._fps = None

    def create_file_for_buffer(self, subfolder: str, bufname: str):
        raise NotImplementedError()

    def init_buffer(self, name: str, max_seconds: int):
        if max_seconds > 0:
            self._buffers[name] = deque(maxlen=int(max_seconds * self._fps))

    def add_to_buffer(self, name: str, frame):
        if self._buffers[name] is None:
            return

        if len(self._buffers[name]) == self._buffers[name].maxlen:
            self._buffers[name].popleft()

        self._buffers[name].append(frame)

    def is_buffer_full(self, name: str) -> bool:
        return len(self._buffers[name]) == self._buffers[name].maxlen

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

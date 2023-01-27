from collections import deque
from pathlib import Path
from typing import Union, Dict

import cv2
import depthai as dai
import numpy as np

from depthai_sdk.recorders.video_writers import AbstractWriter
from depthai_sdk.recorders.video_writers.utils import create_writer_dir


class VideoWriter(AbstractWriter):
    def __init__(self, path: Path, name: str, fourcc: str, fps: float):  # TODO: fourcc is not used
        self.path = path
        self.name = name

        self._fourcc = None
        self._w, self._h = None, None
        self._fps = fps

        self._buffers: Dict[str, deque] = {}
        self._file = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set_fourcc(self, fourcc: str):
        self._fourcc = fourcc

    def create_file(self, subfolder: str, bufname: str):
        if self._buffers[bufname] is None:
            raise RuntimeError(f"Buffer {bufname} is not enabled")

        if len(self._buffers[bufname]) == 0:
            return None

        path_to_file = create_writer_dir(self.path / subfolder, self.name, 'avi')
        frame = self._buffers[bufname][0]
        self._create_file(path_to_file, frame)

    def _create_file(self, path_to_file: Union[Path, str], frame: Union[dai.ImgFrame, np.ndarray]):
        if isinstance(frame, np.ndarray):
            self._h, self._w = frame.shape[:2]
        else:
            self._h, self._w = frame.getHeight(), frame.getWidth()

        # Disparity - RAW8
        # Depth - RAW16
        if self._fourcc is None:
            if isinstance(frame, np.ndarray):
                c = 1 if frame.ndim == 2 else frame.shape[2]
                self._fourcc = "GRAY" if c == 1 else "I420"
            else:
                if frame.getType() == dai.ImgFrame.Type.RAW16:  # Depth
                    self._fourcc = "FFV1"
                elif frame.getType() == dai.ImgFrame.Type.RAW8:  # Mono Cams
                    self._fourcc = "GREY"
                else:
                    self._fourcc = "I420"

        self._file = cv2.VideoWriter(path_to_file,
                               cv2.VideoWriter_fourcc(*self._fourcc),
                               self._fps,
                               (self._w, self._h),
                               isColor=self._fourcc != "GREY")

    def init_buffer(self, name: str, max_seconds: int):
        if max_seconds > 0:
            self._buffers[name] = deque(maxlen=int(max_seconds * self._fps))

    def add_to_buffer(self, name: str, frame: Union[dai.ImgFrame, np.ndarray]):
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

    def write(self, frame: Union[dai.ImgFrame, np.ndarray]):
        if self._file is None:
            path_to_file = create_writer_dir(self.path, self.name, 'avi')  # What if the file already exists?
            self._create_file(path_to_file, frame)
        self._file.write(frame if isinstance(frame, np.ndarray) else frame.getCvFrame())

    def close(self):
        self._file.release()

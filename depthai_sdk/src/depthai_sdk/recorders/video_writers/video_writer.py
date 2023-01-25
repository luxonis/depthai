from collections import deque
from pathlib import Path
from typing import Union

import cv2
import depthai as dai
import numpy as np

from depthai_sdk.recorders.video_writers import AbstractWriter
from depthai_sdk.recorders.video_writers.utils import create_writer_dir


class VideoWriter(AbstractWriter):
    _fps: float
    _path: str

    def __init__(self, path: Path, name: str, fourcc: str, fps: float):
        self._file = None
        self._path = create_writer_dir(path, name, 'avi')
        self._fourcc = None

        self._w, self._h = None, None
        self._fps = fps

        self._buffers = {}

    def create_file(self, path: Path, subfolder: str, filename: str, buf_name: str):
        print('Started saving buffer...')
        if self._buffers[buf_name] is None:
            raise RuntimeError(f"Buffer {buf_name} is not enabled")

        if len(self._buffers[buf_name]) == 0:
            return None

        save_path = create_writer_dir(path / subfolder, filename, 'avi')
        frame = self._buffers[buf_name][0]
        return self._create_file(save_path, frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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

        return cv2.VideoWriter(path_to_file,
                               cv2.VideoWriter_fourcc(*self._fourcc),
                               self._fps,
                               (self._w, self._h),
                               isColor=self._fourcc != "GREY")

    def set_fourcc(self, fourcc: str):
        self._fourcc = fourcc

    def init_buffer(self, buf_name: str, max_seconds: int):
        if max_seconds > 0:
            self._buffers[buf_name] = deque(maxlen=int(max_seconds * self._fps))

    def add_to_buffer(self, buf_name: str, frame: Union[dai.ImgFrame, np.ndarray]):
        if self._buffers[buf_name] is None:
            return

        if len(self._buffers[buf_name]) == self._buffers[buf_name].maxlen:
            self._buffers[buf_name].popleft()  # BEFORE WAS pop()

        self._buffers[buf_name].append(frame)

    def is_buffer_full(self, buf_name: str) -> bool:
        return len(self._buffers[buf_name]) == self._buffers[buf_name].maxlen

    def is_buffer_empty(self, buf_name: str) -> bool:
        return len(self._buffers[buf_name]) == 0

    def write_to_file(self, buf_name: str, file: cv2.VideoWriter):
        if len(self._buffers[buf_name]) > 0:
            el = self._buffers[buf_name].popleft()
            file.write(el if isinstance(el, np.ndarray) else el.getCvFrame())

    def write(self, frame: Union[dai.ImgFrame, np.ndarray]):
        if not self._file:
            self._file = self._create_file(self._path, frame)
        self._file.write(frame if isinstance(frame, np.ndarray) else frame.getCvFrame())

    def close(self):
        if self._file:
            self._file.release()

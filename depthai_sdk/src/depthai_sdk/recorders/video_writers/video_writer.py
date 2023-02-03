from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Union

try:
    import cv2
except ImportError:
    cv2 = None

import depthai as dai
import numpy as np

from depthai_sdk.recorders.video_writers import AbstractWriter
from depthai_sdk.recorders.video_writers.utils import create_writer_dir


class VideoWriter(AbstractWriter):
    _fps: float
    _path: str

    def __init__(self, path: Path, name: str, fourcc: str, fps: float):
        self.file = None
        self._path = create_writer_dir(path, name, 'avi')
        self._fourcc = None

        self._w, self._h = None, None
        self._fps = fps

        self._buffer = None
        self._is_buffer_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init_buffer(self, max_seconds: int):
        if max_seconds > 0:
            self._buffer = deque(maxlen=int(max_seconds * self._fps))
            self._is_buffer_enabled = True

    def _create_file(self, frame: Union[dai.ImgFrame, np.ndarray]):
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

        self.file = cv2.VideoWriter(self._path,
                                    cv2.VideoWriter_fourcc(*self._fourcc),
                                    self._fps,
                                    (self._w, self._h),
                                    isColor=self._fourcc != "GREY")

    def close(self):
        if self.file:
            self.file.release()

    def save_snapshot(self, duration: int, dir_path: Union[Path, str] = None):
        if self._buffer is None:
            raise RuntimeError("Buffer is not enabled")

        if len(self._buffer) == 0:
            return None

        snapshot_name = f'snapshot_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.avi'
        save_path = Path(dir_path or self._path.partition("/")[0], snapshot_name)

        snapshot_file = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*self._fourcc),
            self._fps,
            (self._w, self._h),
            isColor=self._fourcc == "I420"
        )

        # Copy queue
        buffer_copy = self._buffer.copy()

        n_skip_frames = int(self._fps * (self._fps * duration))
        while len(buffer_copy) > 0:
            # Wait til we reach the desired time
            if n_skip_frames > 0:
                n_skip_frames -= 1
                buffer_copy.popleft()
                continue

            el = buffer_copy.popleft()
            snapshot_file.write(el if isinstance(el, np.ndarray) else el.getCvFrame())

        snapshot_file.release()
        print('Snapshot saved to', save_path)

    def set_fourcc(self, fourcc: str):
        self._fourcc = fourcc

    def add_to_buffer(self, frame: Union[dai.ImgFrame, np.ndarray]):
        if not self._is_buffer_enabled:
            return

        if len(self._buffer) == self._buffer.maxlen:
            self._buffer.pop()

        self._buffer.append(frame)

    def write(self, frame: Union[dai.ImgFrame, np.ndarray]):
        if self.file is None:
            self._create_file(frame)

        self.add_to_buffer(frame)

        self.file.write(frame if isinstance(frame, np.ndarray) else frame.getCvFrame())

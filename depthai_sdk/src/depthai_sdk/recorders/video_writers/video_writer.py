import datetime
from collections import deque
from pathlib import Path
from typing import Union

import cv2
import depthai as dai
import numpy as np

from depthai_sdk.recorders.video_writers import AbstractWriter


class VideoWriter(AbstractWriter):
    file = None
    _fps: float
    _path: str

    def __init__(self, folder: Path, name: str, fourcc: str, fps: float, keep_last_seconds: int = 0):
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        self._path = str(folder / f"{name}.avi")
        self._keep_last_seconds = keep_last_seconds
        self._fourcc = None

        self._w, self._h = None, None

        self._buffer = None
        if self._keep_last_seconds > 0:
            self._buffer = deque(maxlen=int(keep_last_seconds * fps))

        self._fps = fps

    def _create_file(self, frame: Union[dai.ImgFrame, np.ndarray]):
        if isinstance(frame, np.ndarray):
            self._h, self._w = frame.shape[:2]
        else:
            self._h, self._w = frame.getHeight(), frame.getWidth()

        # Disparity - RAW8
        # Depth - RAW16

        if isinstance(frame, np.ndarray):
            c = 1 if len(frame.shape) == 2 else frame.shape[2]
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
                                    isColor=self._fourcc == "I420")

    def close(self):
        self.file.release()

    def get_last(self, seconds: float = 0.0):
        if self._buffer is None:
            raise RuntimeError("Buffer is not enabled")

        if len(self._buffer) == 0:
            return None

        snapshot_path = f'{self._path.partition("/")[0]}/snapshot_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.avi'
        snapshot_file = cv2.VideoWriter(
            snapshot_path,
            cv2.VideoWriter_fourcc(*self._fourcc),
            self._fps,
            (self._w, self._h),
            isColor=self._fourcc == "I420"
        )

        # Copy queue
        buffer_copy = self._buffer.copy()

        n_skip_frames = int(self._fps * (self._keep_last_seconds - seconds))
        while len(buffer_copy) > 0:
            # Wait til we reach the desired time
            if n_skip_frames > 0:
                n_skip_frames -= 1
                buffer_copy.popleft()
                continue

            el = buffer_copy.popleft()
            snapshot_file.write(el if isinstance(el, np.ndarray) else el.getCvFrame())

        snapshot_file.release()
        print('Snapshot saved to', snapshot_path)

    def write(self, frame: Union[dai.ImgFrame, np.ndarray]):
        if self.file is None:
            self._create_file(frame)

        # Write to buffer if needed
        if self._buffer is not None:
            if len(self._buffer) == self._buffer.maxlen:
                self._buffer.pop()

            self._buffer.append(frame)

        self.file.write(frame if isinstance(frame, np.ndarray) else frame.getCvFrame())

import logging
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
    """
    Writes raw streams to mp4 using cv2.VideoWriter.
    """
    _fps: float
    _path: str

    def __init__(self, path: Path, name: str, fourcc: str, fps: float):
        """
        Args:
            path: Path to save the output. Either a folder or a file.
            name: Name of the stream.
            fourcc: FourCC code of the codec used to compress the frames.
            fps: Frames per second.
        """
        self.file = None
        self._path = create_writer_dir(path, name, 'mp4')
        if not self._path.endswith('.mp4'):
            self._path = self._path[:-4] + '.mp4'

        self._fourcc = None

        self._w, self._h = None, None
        self._fps = fps

        self._buffer = None
        self._is_buffer_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init_buffer(self, max_seconds: int) -> None:
        """
        Initialize the buffer to store the frames before writing them to the file.

        Args:
            max_seconds: Maximum number of seconds to store in the buffer.
        """
        if max_seconds > 0:
            self._buffer = deque(maxlen=int(max_seconds * self._fps))
            self._is_buffer_enabled = True

    def _create_file(self, frame: Union[dai.ImgFrame, np.ndarray]) -> None:
        """
        Create the file based on the frame size and the codec.

        Args:
            frame: Frame to get the size from.
        """
        if isinstance(frame, np.ndarray):
            self._h, self._w = frame.shape[:2]
        else:
            self._h, self._w = frame.getHeight(), frame.getWidth()

        c = 3  # Default to 3 channels
        if isinstance(frame, np.ndarray):
            c = 1 if frame.ndim == 2 else frame.shape[2]

        self._fourcc = 'mp4v'
        self.file = cv2.VideoWriter(self._path,
                                    cv2.VideoWriter_fourcc(*self._fourcc),
                                    self._fps,
                                    (self._w, self._h),
                                    isColor=c != 1)

    def close(self) -> None:
        """
        Close the file if it is open.
        """
        if self.file:
            self.file.release()

    def add_to_buffer(self, frame: Union[dai.ImgFrame, np.ndarray]) -> None:
        """
        Add a frame to the buffer if it is enabled.

        Args:
            frame: Frame to add to the buffer.
        """
        if not self._is_buffer_enabled:
            return

        if len(self._buffer) == self._buffer.maxlen:
            self._buffer.pop()

        self._buffer.append(frame)

    def write(self, frame: Union[dai.ImgFrame, np.ndarray]) -> None:
        """
        Write a frame to the file. If buffer is enabled, it will be added to the buffer.

        Args:
            frame: Frame to write to the file.
        """
        if self.file is None:
            self._create_file(frame)

        self.add_to_buffer(frame)

        self.file.write(frame if isinstance(frame, np.ndarray) else frame.getCvFrame())

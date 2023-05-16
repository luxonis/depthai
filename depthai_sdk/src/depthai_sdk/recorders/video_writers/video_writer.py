from collections import deque
from pathlib import Path
from typing import Union

try:
    import cv2
except ImportError:
    cv2 = None

import depthai as dai
import numpy as np

from depthai_sdk.recorders.video_writers import BaseWriter
from depthai_sdk.recorders.video_writers.utils import create_writer_dir


class VideoWriter(BaseWriter):
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

        super().__init__(path, name)

        self._fourcc = None
        self._w, self._h = None, None
        self._fps = fps

        self._buffer = None
        self._is_buffer_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init_buffer(self, name: str, max_seconds: int):
        if max_seconds > 0:
            self._buffers[name] = deque(maxlen=int(max_seconds * self._fps))
            self._is_buffer_enabled = True

    def set_fourcc(self, fourcc: str):
        self._fourcc = fourcc

    def create_file_for_buffer(self, subfolder: str, buf_name: str):
        if self._buffers[buf_name] is None:
            raise RuntimeError(f"Buffer {buf_name} is not enabled")

        if len(self._buffers[buf_name]) == 0:
            return None

        frame = self._buffers[buf_name][0]
        self.create_file(subfolder, frame)

    def create_file(self, subfolder: str, frame: Union[dai.ImgFrame, np.ndarray]):
        path_to_file = create_writer_dir(self.path / subfolder, self.name, 'mp4')

        if not path_to_file.endswith('.mp4'):
            path_to_file = path_to_file[:-4] + '.mp4'

        self._create_file(path_to_file, frame)

    def _create_file(self, path_to_file: str, frame: Union[dai.ImgFrame, np.ndarray]):
        if isinstance(frame, np.ndarray):
            self._h, self._w = frame.shape[:2]
        else:
            self._h, self._w = frame.getHeight(), frame.getWidth()

        if not isinstance(frame, np.ndarray):
            frame = frame.getCvFrame()

        c = 1 if frame.ndim == 2 else frame.shape[2]

        self._fourcc = 'mp4v'
        self._file = cv2.VideoWriter(path_to_file,
                                     cv2.VideoWriter_fourcc(*self._fourcc),
                                     self._fps,
                                     (self._w, self._h),
                                     isColor=c != 1)

    def write(self, frame: Union[dai.ImgFrame, np.ndarray]):
        if self._file is None:
            self.create_file(subfolder='', frame=frame)

        self._file.write(frame if isinstance(frame, np.ndarray) else frame.getCvFrame())

    def close(self) -> None:
        """
        Close the file if it is open.
        """
        if self._file:
            self._file.release()
            self._file = None

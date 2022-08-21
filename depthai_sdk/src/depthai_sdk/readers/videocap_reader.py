import array
import cv2
import os
from .abstract_reader import AbstractReader
from typing import List, Tuple

class VideoCapReader(AbstractReader):
    """
    Reads stream from mp4, mjpeg, h264, h265
    """
    initialFrame = None
    shape: Tuple[int, int]

    def __init__(self, source: str) -> None:
        self.reader = cv2.VideoCapture(source)

        ok, self.initialFrame = self.reader.read()
        self.shape = (self.initialFrame.shape[1], self.initialFrame.shape[0])

        file = os.path.basename(source)
        f_name, _ = os.path.splitext(file)
        self._stream = f_name

    def read(self):
        if self.initialFrame is not None:
            f = self.initialFrame.copy()
            self.initialFrame = None
            return f
        if not self.reader.isOpened(): return False
        ok, frame = self.reader.read()
        if not ok: return False
        return frame

    def getStreams(self) -> List[str]:
        return [self._stream]

    def getShape(self, name: str) -> Tuple[int, int]: # Doesn't work as expected!!
        return self.shape

    def close(self):
        self.reader.release()
import array
import cv2
import os
from .abstract_reader import AbstractReader

class VideoCapReader(AbstractReader):
    """
    Reads stream from mp4, mjpeg, h264, h265
    """
    def __init__(self, source: str) -> None:
        self.reader = cv2.VideoCapture(source)

        file = os.path.basename(source)
        f_name, _ = os.path.splitext(file)
        self._stream = f_name

    def read(self):
        if not self.reader.isOpened(): return False
        ok, frame = self.reader.read()
        if not ok: return False
        return frame

    def getStreams(self) -> array:
        return [self._stream]

    def getShape(self, name: str) -> tuple:
        return (
            int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def close(self):
        self.reader.release()
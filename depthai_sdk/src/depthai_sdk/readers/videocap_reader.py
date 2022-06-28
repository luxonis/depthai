import cv2
from .abstract_reader import AbstractReader

class VideoCapReader(AbstractReader):
    def __init__(self, source: str) -> None:
        self.reader = cv2.VideoCapture(source)
    def read(self):
        if not self.reader.isOpened(): return False
        ok, frame = self.reader.read()
        if not ok: return False
        return frame
    def getShape(self) -> tuple:
        return (
            int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    def close(self):
        self.reader.release()
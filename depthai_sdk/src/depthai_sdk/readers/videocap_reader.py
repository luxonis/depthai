import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    import cv2
except ImportError:
    cv2 = None

from depthai_sdk.readers.abstract_reader import AbstractReader

_videoExt = ['.mjpeg', '.avi', '.mp4', '.h265', '.h264']


class VideoCapReader(AbstractReader):
    """
    Reads stream from mp4, mjpeg, h264, h265
    """

    def __init__(self, path: Path) -> None:
        self.initialFrames: Dict[str, Any] = dict()
        self.shapes: Dict[str, Tuple[int, int]] = dict()
        self.readers: Dict[str, cv2.VideoCapture] = dict()

        if path.is_file():
            stream = path.stem if (path.stem in ['left', 'right']) else 'color'
            self.readers[stream] = cv2.VideoCapture(str(path))
        else:
            for fileName in os.listdir(str(path)):
                f_name, ext = os.path.splitext(fileName)
                if ext not in _videoExt: continue
                stream = f_name if (f_name == 'left' or f_name == 'right') else 'color'
                self.readers[stream] = cv2.VideoCapture(str(path / fileName))

        for name, reader in self.readers.items():
            ok, f = reader.read()
            self.shapes[name] = (
                f.shape[1],
                f.shape[0]
            )
            self.initialFrames[name] = f

    def read(self):
        frames = dict()
        for name, reader in self.readers.items():
            if self.initialFrames[name] is not None:
                frames[name] = self.initialFrames[name].copy()
                self.initialFrames[name] = None

            if not self.readers[name].isOpened(): return False

            ok, frame = self.readers[name].read()
            if not ok: return False

            frames[name] = frame

        return frames

    def getStreams(self) -> List[str]:
        return [name for name in self.readers]

    def getShape(self, name: str) -> Tuple[int, int]:  # Doesn't work as expected!!
        return self.shapes[name]

    def close(self):
        [r.release() for _, r in self.readers.items()]

    def disableStream(self, name: str):
        if name in self.readers:
            self.readers.pop(name)

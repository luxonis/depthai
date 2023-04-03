import os
import time
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np

from depthai_sdk.readers.abstract_reader import AbstractReader

# Supported image formats:
_imageExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm',
             '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']


def get_name_flag(name: str) -> Tuple:
    stream = 'color'
    flag = cv2.IMREAD_COLOR
    if name == 'left' or name == 'right':
        flag = cv2.IMREAD_GRAYSCALE
        stream = name
    return (stream, flag)


class ImageReader(AbstractReader):
    """
    Reads image(s) in the file. If file name is 'left'/'right', it will take those as the stereo camera pair, otherwise
    it will send them as color frames. If it has multiple frames of the same stream it will cycle them with 1.5 sec
    delay.
    """

    def __init__(self, path: Path) -> None:
        self.frames: Dict[str, List] = {
            'color': [],
            'left': [],
            'right': [],
        }
        self.cntr: Dict[str, int] = dict()

        if path.is_file():
            stream, flag = get_name_flag(path.stem)
            self.frames[stream].append(cv2.imread(str(path), flag))
        else:
            shape = None
            for fileName in os.listdir(str(path)):
                f_name, ext = os.path.splitext(fileName)
                if ext not in _imageExt: continue
                stream, flag = get_name_flag(f_name)

                frame = cv2.imread(str(path / fileName), flag)
                if shape is None:
                    shape = (frame.shape[1], frame.shape[0])
                # Resize, so all images are the same size
                frame = cv2.resize(frame, shape)

                self.frames[stream].append(frame)

        for name, arr in self.frames.items():
            self.cntr[name] = 0

        self.cycle = time.time()

    def read(self):
        # Increase counters
        if 3 < time.time() - self.cycle:
            self.cycle = time.time()
            for name in self.cntr:
                self.cntr[name] += 1
                if len(self.frames[name]) <= self.cntr[name]:
                    self.cntr[name] = 0

        msgs: Dict[str, np.ndarray] = dict()
        for name, arr in self.frames.items():
            if 0 < len(arr):
                msgs[name] = arr[self.cntr[name]]
        return msgs

    def getStreams(self) -> List[str]:
        streams = []
        for name, arr in self.frames.items():
            if 0 < len(arr):
                streams.append(name)
        return streams

    def getShape(self, name: str) -> Tuple[int, int]:
        shape = self.frames[name][0].shape
        return (shape[1], shape[0])

    def close(self):
        pass

    def disableStream(self, name: str):
        if name in self.frames:
            self.frames[name] = []

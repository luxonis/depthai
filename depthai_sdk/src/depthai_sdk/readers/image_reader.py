import array
import time

import cv2
import os
import numpy as np
from .abstract_reader import AbstractReader
from typing import List, Tuple, Dict
from pathlib import Path

# Supported image formats:
_imageExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm',
             '.pnm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']

class ImageReader(AbstractReader):
    """
    Reads image(s) in the file. If file name is 'left'/'right', it will take those as the stereo camera pair, otherwise
    it will send them as color frames. If it has multiple frames of the same stream it will cycle them with 1.5 sec
    delay.
    """
    frames: Dict[str, List] = {
        'color': [],
        'left': [],
        'right': [],
    }
    cntr: Dict[str, int] = dict()

    def __init__(self, folder: Path) -> None:
        for fileName in os.listdir(str(folder)):
            f_name, ext = os.path.splitext(fileName)
            if ext not in _imageExt: continue

            stream = 'color'
            flag = cv2.IMREAD_COLOR
            if f_name == 'left' or f_name == 'right':
                flag = cv2.IMREAD_GRAYSCALE
                stream = f_name

            self.frames[stream].append(cv2.imread(str(folder / fileName), flag))

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

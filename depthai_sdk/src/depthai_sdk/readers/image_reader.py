import array
import cv2
import os
from .abstract_reader import AbstractReader
from typing import List, Tuple

class ImageReader(AbstractReader):
    """
    Reads the image. Supported image formats:
    bmp, dib, jpeg, jpg, jpe, jp2, png, webp, pbm, pgm, ppm, pxm, pnm, pfm, sr, ras, tiff, tif, exr, hdr, pic 
    """

    def __init__(self, source: str) -> None:
        file = os.path.basename(source)
        f_name, _ = os.path.splitext(file)

        self._stream = 'color'
        flag = cv2.IMREAD_COLOR
        if f_name == 'left' or f_name == 'right':
            flag = cv2.IMREAD_GRAYSCALE
            self._stream = f_name

        self.image = cv2.imread(source, flag)

    def read(self):
        return self.image.copy()

    def getStreams(self) -> List[str]:
        return [self._stream]

    def getShape(self, name: str) -> Tuple[int, int]:
        shape = self.image.shape
        return (shape[1], shape[0])

    def close(self):
        pass
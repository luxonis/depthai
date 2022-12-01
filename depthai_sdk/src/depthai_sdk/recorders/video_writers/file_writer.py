from pathlib import Path

import depthai as dai

from depthai_sdk.recorders.video_writers import AbstractWriter


class FileWriter(AbstractWriter):
    file = None

    def __init__(self, folder: Path, name: str, fourcc: str):
        super().__init__()
        self.file = open(str(folder / f'{name}.dat'), 'wb')

    def close(self):
        self.file.close()

    def get_last(self, seconds: float = 0.0):
        raise NotImplementedError('FileWriter does not support get_last at the moment')

    def write(self, frame: dai.ImgFrame):
        self.file.write(frame.getData())

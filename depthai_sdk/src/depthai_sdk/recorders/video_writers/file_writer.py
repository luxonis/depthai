from pathlib import Path
import depthai as dai

class FileWriter:
    file = None

    def __init__(self, folder: Path, name: str, fourcc: str):
        self.file = open(str(folder / f"{name}.{fourcc}"), 'wb')

    def close(self):
        self.file.close()


    def write(self, frame: dai.ImgFrame):
        self.file.write(frame.getData())
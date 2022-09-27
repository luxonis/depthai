
from pathlib import Path
from .abstract_recorder import Recorder
import depthai as dai
from ..record import Codec

class RawRecorder(Recorder):
    _closed = False
    def __init__(self, folder: Path, codec: Codec):
        self.folder = folder
        self.ext = Codec.fourcc(codec)
        self.files = {}

    def write(self, name: str, frame: dai.ImgFrame):
        if name not in self.files:
            self.__create_file(name)

        self.files[name].write(frame.getCvFrame())

    def __create_file(self, name):
        self.files[name] = open(str(self.folder / f"{name}.{self.ext}"), 'wb')
        # if name == "color": fourcc = "I420"
        # elif name == "depth": fourcc = "Y16 " # 16-bit uncompressed greyscale image
        # else : fourcc = "GREY" #Simple, single Y plane for monochrome images.
        # files[name] = VideoWriter(str(path / f"{name}.avi"), VideoWriter_fourcc(*fourcc), fps, sizes[name], isColor=name=="color")

    def close(self):
        if self._closed: return
        self._closed = True
        print("raw encoded stream(s) saved at:", str(self.folder))
        # Close opened files
        for name in self.files:
            self.files[name].close()

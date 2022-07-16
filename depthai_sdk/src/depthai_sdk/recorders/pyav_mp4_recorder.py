import av
from fractions import Fraction
import time
from pathlib import Path
from .abstract_recorder import Recorder
import depthai as dai

class PyAvRecorder(Recorder):
    _closed = False
    def __init__(self, folder: Path, quality, rgbFps: float, monoFps: float):
        self.folder = folder
        # codec could also be "h264", but it's not (yet) supported
        self.codec = "hevc" if int(quality) == 4 else "mjpeg"
        self._monoFps = monoFps
        self._rgbFps = rgbFps
        self.start = dict()

        self.files = {}

    def write(self, name: str, frame: dai.ImgFrame):
        if name not in self.files:
            self.__create_file(name)

        packet = av.Packet(frame.getData()) # Create new packet with byte array
        
        # Set frame timestamp
        if name not in self.start:
            self.start[name] = frame.getTimestampDevice()

        ts = int((frame.getTimestampDevice() - self.start[name]).total_seconds() * 1e6) # To microsec
        packet.dts = ts
        packet.pts = ts

        self.files[name].mux_one(packet) # Mux the Packet into container

    def __create_file(self, name):
        self.files[name] = av.open(str(self.folder / f"{name}.mp4"), 'w')
        fps = self._rgbFps if name == "color" else self._monoFps
        stream = self.files[name].add_stream(self.codec, rate=int(fps))
        stream.time_base = Fraction(1, 1000 * 1000) # Microseconds

        # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
        if self.codec == "mjpeg":
            stream.pix_fmt = "yuvj420p"

    def close(self):
        if self._closed: return
        self._closed = True
        print(".mp4 container(s) saved at:", str(self.folder))
        # Close the containers
        for name in self.files:
            self.files[name].close()

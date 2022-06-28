import av
from fractions import Fraction
import time
from pathlib import Path
from .abstract_recorder import Recorder

class PyAvRecorder(Recorder):
    closed = False
    def __init__(self, folder: Path, quality, fps: int):
        print('quality',quality)
        self.folder = folder
        # codec could also be "h264", but it's not (yet) supported
        self.codec = "hevc" if int(quality) == 4 else "mjpeg"
        self.fps = fps
        self.start = None

        self.files = {}

    def write(self, name, frame):
        if name not in self.files:
            self.__create_file(name)

        packet = av.Packet(frame) # Create new packet with byte array
        # Set frame timestamp
        packet.pts = int((time.time() - self.start) * 1000 * 1000)

        self.files[name].mux_one(packet) # Mux the Packet into container

    def __create_file(self, name):
        self.files[name] = av.open(str(self.folder / f"{name}.mp4"), 'w')
        stream = self.files[name].add_stream(self.codec, rate=self.fps)
        stream.time_base = Fraction(1, 1000 * 1000) # Microseconds

        # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
        if self.codec == "mjpeg":
            stream.pix_fmt = "yuvj420p"

        if self.start is None:
            self.start = time.time()

    def close(self):
        if self.closed: return
        self.closed = True
        # Close the containers
        for name in self.files:
            self.files[name].close()

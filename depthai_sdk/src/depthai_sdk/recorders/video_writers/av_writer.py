from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import av
import depthai as dai

from depthai_sdk.recorders.video_writers import AbstractWriter


class AvWriter(AbstractWriter):
    start_ts: timedelta = None
    file = None

    def __init__(self, folder: Path, name: str, fourcc: str, fps: float):
        self.start_ts = None
        self.file = av.open(str(folder / f"{name}.mp4"), 'w')

        stream = self.file.add_stream(fourcc, rate=int(fps))
        stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds

        # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
        if fourcc == "mjpeg":
            stream.pix_fmt = "yuvj420p"

    def close(self):
        self.file.close()

    def write(self, frame: dai.ImgFrame):
        packet = av.Packet(frame.getData())  # Create new packet with byte array

        # Set frame timestamp
        if self.start_ts is None:
            self.start_ts = frame.getTimestampDevice()

        ts = int((frame.getTimestampDevice() - self.start_ts).total_seconds() * 1e6)  # To microsec
        packet.dts = ts
        packet.pts = ts

        self.file.mux_one(packet)  # Mux the Packet into container

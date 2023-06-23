from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import depthai as dai

from depthai_sdk.recorders.video_writers import BaseWriter
from depthai_sdk.recorders.video_writers.utils import create_writer_dir


class AvWriter(BaseWriter):
    def __init__(self, path: Path, name: str, fourcc: str, fps: float):
        super().__init__(path, name)

        self.start_ts = None
        self._fps = fps
        self._fourcc = fourcc

    def _create_stream(self, fourcc, fps) -> None:
        """Create stream in file with given fourcc and fps, works in-place."""
        stream = self._file.add_stream(fourcc, rate=int(fps))
        stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds

        # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
        if fourcc == 'mjpeg':
            stream.pix_fmt = 'yuvj420p'

    def create_file_for_buffer(self, subfolder: str, buf_name: str):  # independent of type of frames
        self.create_file(subfolder)

    def create_file(self, subfolder: str):
        path_to_file = create_writer_dir(self.path / subfolder, self.name, 'mp4')
        self._create_file(path_to_file)

    def _create_file(self, path_to_file: str):
        global av
        import av as av
        self._file = av.open(path_to_file, 'w')
        self._create_stream(self._fourcc, self._fps)

    def write(self, frame: dai.ImgFrame) -> None:
        if self._file is None:
            self.create_file(subfolder='')

        packet = av.Packet(frame.getData())  # Create new packet with byte array

        # Set frame timestamp
        if self.start_ts is None:
            self.start_ts = frame.getTimestampDevice()

        ts = int((frame.getTimestampDevice() - self.start_ts).total_seconds() * 1e6)  # To microsec
        packet.dts = ts
        packet.pts = ts

        self._file.mux_one(packet)  # Mux the Packet into container

    def close(self):
        if self._file is not None:
            self._file.close()

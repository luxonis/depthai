from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import depthai as dai

from depthai_sdk.recorders.video_writers import AbstractWriter
from depthai_sdk.recorders.video_writers.utils import create_writer_dir


class AvWriter(AbstractWriter):
    start_ts: timedelta = None
    file = None

    def __init__(self, folder: Path, name: str, fourcc: str, fps: float):
        global av
        import av as av

        name = create_writer_dir(folder, name, 'mp4')

        self.start_ts = None
        self.folder = folder
        self.file = av.open(name, 'w')
        self._fps = fps
        self._fourcc = fourcc

        self._create_stream(self.file, fourcc, fps)

        # self._buffer = None
        # if self._keep_last_seconds > 0:
        #     self._buffer = deque(maxlen=int(10 * fps))

    def _create_stream(self, file, fourcc, fps) -> None:
        """Create stream in file with given fourcc and fps, works in-place."""
        stream = file.add_stream(fourcc, rate=int(fps))
        stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds

        # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
        if fourcc == 'mjpeg':
            stream.pix_fmt = 'yuvj420p'

    def write(self, frame: dai.ImgFrame) -> None:
        packet = av.Packet(frame.getData())  # Create new packet with byte array

        # if self._buffer is not None:
        #     if len(self._buffer) == self._buffer.maxlen:
        #         self._buffer.pop()
        #
        #     self._buffer.append(frame)

        # Set frame timestamp
        if self.start_ts is None:
            self.start_ts = frame.getTimestampDevice()

        ts = int((frame.getTimestampDevice() - self.start_ts).total_seconds() * 1e6)  # To microsec
        packet.dts = ts
        packet.pts = ts

        self.file.mux_one(packet)  # Mux the Packet into container

    # def get_last(self, seconds: float = 0.0):
    #     if self._buffer is None:
    #         print('Buffer is not initialized, cannot get last frames')
    #         return
    #
    #     if len(self._buffer) == 0:
    #         print('Buffer is empty, cannot get last frames')
    #         return
    #
    #     snapshot_path = f'{self.folder}/snapshot_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
    #     snapshot_file = av.open(snapshot_path, 'w')
    #     self._create_stream(snapshot_file, self._fourcc, self._fps)
    #
    #     # Copy queue
    #     buffer_copy = self._buffer.copy()
    #
    #     start_ts = None
    #     n_skip_frames = int(self._fps * (self._keep_last_seconds - seconds))
    #     while len(buffer_copy) > 0:
    #         # Wait til we reach the desired time
    #         if n_skip_frames > 0:
    #             n_skip_frames -= 1
    #             buffer_copy.popleft()
    #             continue
    #
    #         el = buffer_copy.popleft()
    #         packet = av.Packet(el.getData())
    #         if start_ts is None:
    #             start_ts = el.getTimestampDevice()
    #
    #         ts = int((el.getTimestampDevice() - start_ts).total_seconds() * 1e6)  # To microsec
    #         packet.dts = ts
    #         packet.pts = ts
    #
    #         snapshot_file.mux(packet)
    #
    #     snapshot_file.close()
    #     print('Snapshot saved to', snapshot_path)

    def close(self):
        self.file.close()

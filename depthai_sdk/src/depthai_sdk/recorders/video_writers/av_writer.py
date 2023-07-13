import os
from fractions import Fraction
from pathlib import Path
from typing import Tuple, Union

import depthai as dai
import numpy as np

from depthai_sdk.recorders.video_writers.utils import create_writer_dir
from .base_writer import BaseWriter

START_CODE_PREFIX = b"\x00\x00\x01"
KEYFRAME_NAL_TYPE = 5
NAL_TYPE_BITS = 5  # nal unit type is encoded in lower 5 bits


def is_keyframe(encoded_frame: np.array) -> bool:
    """
    Check if encoded frame is a keyframe.
    Args:
        encoded_frame: Encoded frame.

    Returns:
        True if encoded frame is a keyframe, False otherwise.
    """
    byte_stream = bytearray(encoded_frame)
    size = len(byte_stream)

    pos = 0
    while pos < size:
        retpos = byte_stream.find(START_CODE_PREFIX, pos)
        if retpos == -1:
            return False

        # Skip start code
        pos = retpos + 3

        # Extract the first 5 bits
        type_ = byte_stream[pos] >> 3

        if type_ == KEYFRAME_NAL_TYPE:
            return True

    return False


class AvWriter(BaseWriter):
    def __init__(self, path: Path, name: str, fourcc: str, fps: float, frame_shape: Tuple[int, int]):
        """
        Args:
            path: Path to the folder where the file will be created.
            name: Name of the file without extension.
            fourcc: Stream codec.
            fps: Frames per second of the stream.
            frame_shape: Width and height of the frames.
        """
        super().__init__(path, name)

        self.start_ts = None
        self.frame_shape = frame_shape

        self._fps = fps
        self._fourcc = fourcc
        self._stream = None

    def _create_stream(self, fourcc: str, fps: float) -> None:
        """
        Create stream in file with given fourcc and fps, works in-place.

        Args:
            fourcc: Stream codec.
            fps: Frames per second of the stream.
        """
        self._stream = self._file.add_stream(fourcc, rate=int(fps))
        self._stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds

        # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
        if fourcc == 'mjpeg':
            self._stream.pix_fmt = 'yuvj420p'

        if self.frame_shape is not None:
            self._stream.width = self.frame_shape[0]
            self._stream.height = self.frame_shape[1]

    def create_file_for_buffer(self, subfolder: str, buf_name: str) -> None:  # independent of type of frames
        self.create_file(subfolder)

    def create_file(self, subfolder: str) -> None:
        """
        Create file for writing frames.

        Args:
            subfolder: Subfolder relative to the main folder where the file will be created.
        """
        path_to_file = create_writer_dir(self.path / subfolder, self.name, 'mp4')
        self._create_file(path_to_file)

    def _create_file(self, path_to_file: str) -> None:
        """
        Create av container for writing frames.

        Args:
            path_to_file: Path to the file.
        """
        global av
        import av
        self._file = av.open(str(Path(path_to_file).with_suffix(f'.{self._fourcc}')), 'w')
        self._create_stream(self._fourcc, self._fps)

    def write(self, frame: dai.ImgFrame) -> None:
        """
        Write packet bytes to h264 file.

        Args:
            frame: ImgFrame from depthai pipeline.
        """
        if self._file is None:
            self.create_file(subfolder='')

        frame_data = frame.getData()

        if self.start_ts is None and not is_keyframe(frame_data):
            return

        packet = av.Packet(frame_data)  # Create new packet with byte array

        # Set frame timestamp
        if self.start_ts is None:
            self.start_ts = frame.getTimestampDevice()

        ts = int((frame.getTimestampDevice() - self.start_ts).total_seconds() * 1e6)  # To microsec
        packet.dts = ts
        packet.pts = ts
        self._file.mux_one(packet)  # Mux the Packet into container

    def close(self) -> None:
        """
        Close the file and remux it to mp4.
        """
        if self._file is not None:
            p = self._stream.encode(None)
            self._file.mux(p)
            self._file.close()

        # Remux the stream to finalize the output file
        self.remux_video(str(self._file.name))

    def remux_video(self, input_file: Union[Path, str]) -> None:
        """
        Remuxes h264 file to mp4.

        Args:
            input_file: path to h264 file.
        """

        mp4_file = str(Path(input_file).with_suffix('.mp4'))

        if input_file == mp4_file:
            mp4_file = str(Path(input_file).with_suffix('.remuxed.mp4'))

        with av.open(mp4_file, "w", format="mp4") as output_container, \
                av.open(input_file, "r", format=self._fourcc) as input_container:
            input_stream = input_container.streams[0]
            output_stream = output_container.add_stream(template=input_stream, rate=self._fps)

            if self.frame_shape:
                output_stream.width = self.frame_shape[0]
                output_stream.height = self.frame_shape[1]

            frame_time = (1 / self._fps) * input_stream.time_base.denominator
            for i, packet in enumerate(input_container.demux(input_stream)):
                packet.dts = i * frame_time
                packet.pts = i * frame_time
                packet.stream = output_stream
                output_container.mux_one(packet)

        os.remove(input_file)

        if Path(mp4_file).suffix == '.remuxed.mp4':
            os.rename(mp4_file, input_file)

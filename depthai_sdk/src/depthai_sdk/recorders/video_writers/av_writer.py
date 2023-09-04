import os
from fractions import Fraction
from pathlib import Path
from typing import Tuple, Union, Optional, List

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
    def __init__(self, path: Path, name: str, fourcc: str):
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
        self._fourcc = fourcc

        self._stream = None
        self._file = None
        self.closed = False
        self._codec = None  # Used to determine dimensions of encoded frames
        self._frame_buffer: List[dai.ImgFrame] = []

    def _create_stream(self, shape: Tuple) -> None:
        """
        Create stream in file with given fourcc and fps, works in-place.

        Args:
            fourcc: Stream codec.
            fps: Frames per second of the stream.
        """
        self._stream = self._file.add_stream(self._fourcc)
        self._stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds

        # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
        if self._fourcc == 'mjpeg':
            self._stream.pix_fmt = 'yuvj420p'

        self._stream.width = shape[0]
        self._stream.height = shape[1]

    def get_dimension(self, img: dai.ImgFrame) -> Optional[Tuple[int, int]]:
        enc_packets = self._codec.parse(img.getData())
        if len(enc_packets) == 0:
            return None
        frames = self._codec.decode(enc_packets[-1])
        if not frames:
            return None
        return frames[0].width, frames[0].height

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
        # We will remux .h264 later
        suffix = '.h264' if self._fourcc.lower() == 'h264' else '.mp4'
        self._file = av.open(str(Path(path_to_file).with_suffix(suffix)), 'w')

        # Needed to get dimensions from the frame. Only decode first frame.
        self._codec = av.CodecContext.create(self._fourcc, "r")

    def __mux_imgframe(self, frame: dai.ImgFrame) -> None:
        frame_data = frame.getData()

        if self.start_ts is None:
            # For H26x, wait for a keyframe
            if self._fourcc != 'mjpeg' and not is_keyframe(frame_data):
                return

        packet = av.Packet(frame_data)  # Create new packet with byte array

        # Set frame timestamp
        if self.start_ts is None:
            self.start_ts = frame.getTimestampDevice()

        ts = int((frame.getTimestampDevice() - self.start_ts).total_seconds() * 1e6)  # To microsec
        packet.dts = ts + 1  # +1 to avoid zero dts
        packet.pts = ts + 1
        packet.stream = self._stream
        self._file.mux_one(packet)  # Mux the Packet into container

    def write(self, frame: dai.ImgFrame) -> None:
        """
        Write packet bytes to h264 file.

        Args:
            frame: ImgFrame from depthai pipeline.
        """
        if self.closed:
            return
        if self._file is None:
            self.create_file(subfolder='')

        if self._stream is None:
            shape = self.get_dimension(frame)
            if shape is None:
                # Save frame, so we can mux it later when dimnesions are known
                self._frame_buffer.append(frame)
                return

            self._create_stream(shape)
            for buffered_frame in self._frame_buffer:
                self.__mux_imgframe(buffered_frame)

        self.__mux_imgframe(frame)

    def close(self) -> None:
        """
        Close the file and potentially remux it to mp4.
        """
        self.closed = True
        if self._file is not None:
            p = self._stream.encode(None)
            self._file.mux(p)
            self._file.close()

        # Remux the h264 stream to finalize the output file
        if self._fourcc == 'h264':
            self.remux_h264_video(str(self._file.name))

    def remux_h264_video(self, input_file: Union[Path, str]) -> None:
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
            fps = input_stream.average_rate
            output_stream = output_container.add_stream(template=input_stream, rate=fps)

            output_stream.width = input_stream.width
            output_stream.height = input_stream.height

            frame_time = (1 / fps) * input_stream.time_base.denominator
            for i, packet in enumerate(input_container.demux(input_stream)):
                packet.dts = i * frame_time
                packet.pts = i * frame_time
                packet.stream = output_stream
                output_container.mux_one(packet)

        os.remove(input_file)

        if Path(mp4_file).suffix == '.remuxed.mp4':
            os.rename(mp4_file, input_file)

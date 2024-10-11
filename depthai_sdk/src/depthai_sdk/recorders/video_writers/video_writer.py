from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import av
import depthai as dai

from depthai_sdk.recorders.video_writers import BaseWriter
from depthai_sdk.recorders.video_writers.utils import create_writer_dir


class VideoWriter(BaseWriter):
    """
    Writes raw streams to file
    """

    def __init__(self, path: Path, name: str, lossless: bool = False):
        """
        Args:
            path: Path to save the output. Either a folder or a file.
            name: Name of the stream.
            lossless: If True, save the stream without compression.
        """

        super().__init__(path, name)

        self._lossless = lossless

        self._fourcc: str = None
        self._format: str = None
        self._start_ts: timedelta = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_file_for_buffer(self, subfolder: str, buf_name: str):
        if self._buffers[buf_name] is None:
            raise RuntimeError(f"Buffer {buf_name} is not enabled")

        if len(self._buffers[buf_name]) == 0:
            return None

        frame = self._buffers[buf_name][0]
        self.create_file(subfolder, frame)

    def create_file(self, subfolder: str, frame: dai.ImgFrame):
        if self._lossless or frame.getType() == dai.ImgFrame.Type.RAW16:
            extension = 'avi'
        else:
            extension = 'mp4'

        path_to_file = create_writer_dir(self.path / subfolder, self.name, extension)

        if not path_to_file.endswith('.' + extension):
            path_to_file = path_to_file[:-4] + '.' + extension

        self._create_file(path_to_file, frame)

    def _create_file(self, path_to_file: str, frame: dai.ImgFrame):
        options = {}
        if self._lossless:
            self._fourcc = 'rawvideo'
        elif frame.getType() == dai.ImgFrame.Type.RAW16: # Depth map
            self._fourcc = 'ffv1'
            self._format = 'gray16le'
        else:  # Mono/Color, encode
            self._fourcc = 'h264'
            options['crf'] = '15'

        self._file = av.open(path_to_file, 'w')
        self._stream = self._file.add_stream(self._fourcc)
        self._stream.options = options
        self._stream.time_base = Fraction(1, 1000)
        self._stream.codec_context.width = frame.getWidth()
        self._stream.codec_context.height = frame.getHeight()

        if self._fourcc == 'ffv1':
            self._stream.width = frame.getWidth()
            self._stream.height = frame.getHeight()
            self._stream.pix_fmt = 'gray16le' # Required for depth recording to work correctly

    def write(self, img_frame: dai.ImgFrame):
        if self._file is None:
            self.create_file(subfolder='', frame=img_frame)
        if self._start_ts is None:
            self._start_ts = img_frame.getTimestampDevice()

        if img_frame.getType() == dai.ImgFrame.Type.YUV420p:
            video_format = 'yuv420p'
        elif img_frame.getType() == dai.ImgFrame.Type.NV12:
            video_format = 'nv12'
        elif img_frame.getType() in [dai.ImgFrame.Type.RAW8, dai.ImgFrame.Type.GRAY8]:
            video_format = 'gray'
        elif img_frame.getType() == dai.ImgFrame.Type.RAW16:
            video_format = 'gray16le'
        else:
            raise ValueError(f'Unsupported frame type: {img_frame.getType()}')
        video_frame = av.VideoFrame.from_ndarray(img_frame.getFrame(), format=video_format)

        ts = int((img_frame.getTimestampDevice() - self._start_ts).total_seconds() * 1e3)  # To milliseconds
        video_frame.pts = ts + 1

        for packet in self._stream.encode(video_frame):
            self._file.mux(packet)

    def close(self) -> None:
        """
        Close the file if it is open.
        """
        if self._file:
            # Flush stream
            for packet in self._stream.encode():
                self._file.mux(packet)

            # Close output file
            self._file.close()

from typing import Union, Dict

import numpy as np

from .abstract_recorder import *
from depthai_sdk.logger import LOGGER


class VideoRecorder(Recorder):
    """
    Writes video streams (.mjpeg/.h264/.hevc) or directly to mp4/avi container.
    """

    def __init__(self, lossless: bool = False):
        self.path = None
        self._stream_type = dict()
        self._writers = dict()
        self._closed = False
        self._lossless = lossless

    def __getitem__(self, item):
        return self._writers[item]

    # TODO device is not used
    def update(self, path: Path, device: dai.Device, xouts: List['XoutFrames']):
        """
        Update the recorder with new streams.
        Args:
            path: Path to save the output. Either a folder or a file.
            device: Device to get the streams from.
            xouts: List of output streams.
        """
        if path is None:
            return

        self.path = path
        if path.suffix == '' and path != Path('.'):  # If no extension, create a folder
            self.path.mkdir(parents=True, exist_ok=True)

        for xout in xouts:
            # for example, 'color_bitstream' (encoded) or 'color_video' (unencoded),
            # if component was created with name='color'
            xout_name = xout.name  # for example, 'color' --> file is color.mp4 (encoded) or color.avi (unencoded)
            file_name = xout_name
            if file_name.startswith('CameraBoardSocket.'):
                file_name = file_name[len('CameraBoardSocket.'):]
            stream = OakStream(xout)
            fourcc = stream.fourcc()  # TODO add default fourcc? stream.fourcc() can be None.

            print(fourcc, xout_name, stream.type)
            if stream.is_raw() or stream.is_depth():
                from .video_writers.video_writer import VideoWriter
                self._writers[xout_name] = VideoWriter(self.path, file_name, self._lossless)
            else:
                try:
                    from .video_writers.av_writer import AvWriter
                    self._writers[xout_name] = AvWriter(self.path, file_name, fourcc)
                except Exception as e:
                    # TODO here can be other errors, not only import error
                    LOGGER.warning(f'Exception while creating AvWriter: {e}.'
                                    '\nFalling back to FileWriter, saving uncontainerized encoded streams.')
                    from .video_writers.file_writer import FileWriter
                    self._writers[xout_name] = FileWriter(self.path, file_name, fourcc)

    def create_files_for_buffer(self, subfolder: str, buf_name: str):
        for _, writer in self._writers.items():
            writer.create_file_for_buffer(subfolder, buf_name)

    def create_file_for_buffer(self, wr_name: str, subfolder: str, buf_name: str):
        self._writers[wr_name].create_file_for_buffer(subfolder, buf_name)

    def create_file(self, wr_name: str, subfolder: str, frame: Union[np.ndarray, dai.ImgFrame]):
        self._writers[wr_name].create_file(subfolder, frame)

    def init_buffers(self, buffers: Dict[str, int]):
        for _, writer in self._writers.items():
            for name, max_seconds in buffers.items():
                writer.init_buffer(name, max_seconds)

    def add_to_buffers(self, buf_name: str, frames: Dict[str, Union[np.ndarray, dai.ImgFrame]]):
        for name, writer in self._writers.items():
            writer.add_to_buffer(buf_name, frames[name])

    def add_to_buffer(self, wr_name: str, buf_name: str, frame: Union[np.ndarray, dai.ImgFrame]):
        self._writers[wr_name].add_to_buffer(buf_name, frame)

    def is_buffer_full(self, wr_name: str, buf_name: str):
        return self._writers[wr_name].is_buffer_full(buf_name)

    def is_buffer_empty(self, wr_name: str, buf_name: str):
        return self._writers[wr_name].is_buffer_empty(buf_name)

    def write_from_buffer(self, wr_name: str, buf_name: str, n_elems: int):
        self._writers[wr_name].write_from_buffer(buf_name, n_elems)

    def write(self, name: str, frame: Union[np.ndarray, dai.ImgFrame]):
        self._writers[name].write(frame)

    def close_files(self):
        for _, writer in self._writers.items():
            writer.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        LOGGER.info(f'Video Recorder saved stream(s) to folder: {str(self.path)}')
        # Close opened files
        for name, writer in self._writers.items():
            writer.close()

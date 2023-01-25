from typing import Union

import numpy as np

from .abstract_recorder import *


class VideoRecorder(Recorder):
    """
    Writes encoded streams raw (.mjpeg/.h264/.hevc) or directly to mp4 container.
    Writes unencoded streams to mp4 using cv2.VideoWriter
    """

    def __init__(self):
        self.path = None
        self._stream_type = dict()
        self._writers = dict()
        self._closed = False

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
            name = xout.name
            stream = OakStream(xout)
            fourcc = stream.fourcc()  # TODO add default fourcc? stream.fourcc() can be None.
            if stream.is_raw():
                from .video_writers.video_writer import VideoWriter
                self._writers[name] = VideoWriter(self.path, name, fourcc, xout.fps)
            else:
                try:
                    from .video_writers.av_writer import AvWriter
                    self._writers[name] = AvWriter(self.path, name, fourcc, xout.fps)
                except Exception as e:
                    # TODO here can be other errors, not only import error
                    print('Exception while creating AvWriter: ', e)
                    print('Falling back to FileWriter, saving uncontainerized encoded streams.')
                    from .video_writers.file_writer import FileWriter
                    self._writers[name] = FileWriter(self.path, name, fourcc)

    def write(self, name: str, frame: Union[np.ndarray, dai.ImgFrame]):
        self._writers[name].write(frame)

    def add_to_buffer(self, name: str, frame: Union[np.ndarray, dai.ImgFrame]):
        self._writers[name].add_to_buffer(frame)

    def close(self):
        if self._closed: return
        self._closed = True
        print("Video Recorder saved stream(s) to folder:", str(self.path))
        # Close opened files
        for name, writer in self._writers.items():
            writer.close()

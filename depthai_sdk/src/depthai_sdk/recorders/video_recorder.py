from pathlib import Path
from typing import List, Dict, Any

import depthai as dai

from depthai_sdk.oak_outputs.xout import XoutFrames
from .abstract_recorder import *


class VideoRecorder(Recorder):
    """
    Writes encoded streams raw (.mjpeg/.h264/.hevc) or directly to mp4 container.
    Writes unencoded streams to mp4 using cv2.VideoWriter
    """
    _closed = False
    _writers: Dict[str, Any]

    def update(self, path: Path, _, xouts: List[XoutFrames]):
        self.folder = path
        self._stream_type = dict()
        self._writer = dict()

        for xout in xouts:
            name = xout.frames.friendly_name or xout.frames.name
            stream = OakStream(xout)
            if stream.isRaw():
                from .video_writers.video_writer import VideoWriter
                self._writer[name] = VideoWriter(self.folder, name, stream.fourcc(), xout.fps)
            else:
                try:
                    from .video_writers.av_writer import AvWriter
                    self._writer[name] = AvWriter(self.folder, name, stream.fourcc(), xout.fps)
                except:
                    print("'av' library is not installed, depthai-record will save uncontainerized encoded streams.")
                    from .video_writers.file_writer import FileWriter
                    self._writer[name] = FileWriter(self.folder, name, stream.fourcc())

    def write(self, name: str, frame: dai.ImgFrame):
        self._writer[name].write(frame)

    def close(self):
        if self._closed: return
        self._closed = True
        print("Video Recorder saved stream(s) to folder:", str(self.folder))
        # Close opened files
        for name, writer in self._writer.items():
            writer.close()

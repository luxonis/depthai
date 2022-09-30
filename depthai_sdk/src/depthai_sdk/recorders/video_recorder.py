
from pathlib import Path
from .abstract_recorder import Recorder
import depthai as dai
from ..record import OakStream
from typing import List, Dict, Any
from ..oak_outputs.xout import XoutFrames




class VideoRecorder(Recorder):
    """
    Writes encoded streams raw (.mjpeg/.h264/.hevc) or directly to mp4 container.
    Writes unencoded streams to mp4 using cv2.VideoWriter
    """
    _closed = False
    _writers: Dict[str, Any]

    def __init__(self, folder: Path, xouts: List[XoutFrames]):

        self.folder = folder

        self._stream_type = dict()

        self._writer = dict()
        for xout in xouts:
            name = xout.frames.name
            stream = OakStream(xout)
            if stream.isRaw():
                from .video_writers.video_writer import VideoWriter
                self._writer[name] = VideoWriter(folder, name, stream.fourcc(), xout.fps)
            else:
                try:
                    from .video_writers.av_writer import AvWriter
                    self._writer[name] = AvWriter(folder, name, stream.fourcc(), xout.fps)
                except:
                    print("'av' library is not installed, depthai-record will save uncontainerized encoded streams.")
                    from .video_writers.file_writer import FileWriter
                    self._writer[name] = FileWriter(folder, name, stream.fourcc())


    def write(self, name: str, frame: dai.ImgFrame):
        self._writer[name].write(frame)

    def close(self):
        if self._closed: return
        self._closed = True
        print("Video Recorder saved stream(s) to folder:", str(self.folder))
        # Close opened files
        for name, writer in self._writer.items():
            writer.close()

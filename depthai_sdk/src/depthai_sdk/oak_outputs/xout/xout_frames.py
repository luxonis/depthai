from typing import Tuple, List

import numpy as np

from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.recorders.video_writers import AvWriter
from depthai_sdk.visualize.configs import TextPosition
from depthai_sdk.visualize.visualizer import Platform, Visualizer

try:
    import cv2
except ImportError:
    cv2 = None


class XoutFrames(XoutBase):
    """
    Single message, no syncing required
    """

    def __init__(self, frames: StreamXout, fps: float = 30, frame_shape: Tuple[int, ...] = None):
        self.frames = frames
        self.name = frames.name

        self.fps = fps
        self._video_recorder = None
        self._is_recorder_enabled = None
        self._frame_shape = frame_shape

        super().__init__()

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        self._visualizer = visualizer
        self._visualizer_enabled = visualizer_enabled
        self.name = name or self.name

    def setup_recorder(self,
                       recorder: VideoRecorder,
                       encoding: str = 'mp4v'):
        self._video_recorder = recorder
        # Enable encoding for the video recorder
        self._video_recorder[self.name].set_fourcc(encoding)

    def visualize(self, packet: FramePacket) -> None:
        """
        Called from main thread if visualizer is not None.
        """
        # Frame shape may be 1D, that means it's an encoded frame
        if self._visualizer.frame_shape is None or np.array(self._visualizer.frame_shape).ndim == 1:
            if self._frame_shape is not None:
                self._visualizer.frame_shape = self._frame_shape
            else:
                self._visualizer.frame_shape = packet.frame.shape

        if self._visualizer.config.output.show_fps:
            self._visualizer.add_text(
                text=f'FPS: {self._fps.fps():.1f}',
                position=TextPosition.TOP_LEFT
            )

        if self.callback:  # Don't display frame, call the callback
            self.callback(packet)
        else:
            packet.frame = self._visualizer.draw(packet.frame)
            # Draw on the frame
            if self._visualizer.platform == Platform.PC:
                cv2.imshow(self.name, packet.frame)
            else:
                pass

    def on_record(self, packet) -> None:
        if self._video_recorder:
            # TODO not ideal to check it this way
            if isinstance(self._video_recorder[self.name], AvWriter):
                self._video_recorder.write(self.name, packet.imgFrame)
            else:
                self._video_recorder.write(self.name, packet.frame)
        # else:
        #     self._video_recorder.add_to_buffer(self.name, packet.frame)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames]

    def new_msg(self, name: str, msg) -> None:
        if name not in self._streams:
            return

        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        packet = FramePacket(name,
                             msg,
                             msg.getCvFrame() if cv2 else None,
                             self._visualizer)

        self.queue.put(packet, block=False)

    def __del__(self):
        if self._video_recorder:
            self._video_recorder.close()

from typing import List
from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout

try:
    import cv2
except ImportError:
    cv2 = None


class XoutFrames(XoutBase):
    """
    Stream of frames. Single message, no syncing required.
    """

    def __init__(self, frames: StreamXout):
        """
        Args:
            frames: StreamXout object.
            fps: Frames per second for the output stream.
            frame_shape: Shape of the frame. If not provided, it will be inferred from the first frame.
        """
        self.frames = frames
        self.name = frames.name

        super().__init__()

    def xstreams(self) -> List[StreamXout]:
        return [self.frames]

    def new_msg(self, name: str, msg):
        if name not in self._streams:
            return
        return FramePacket(self.get_packet_name(), msg)

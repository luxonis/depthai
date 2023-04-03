from typing import Tuple

import numpy as np

from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames

try:
    import cv2
except ImportError:
    cv2 = None


class XoutMjpeg(XoutFrames):
    name: str = "MJPEG Stream"

    def __init__(self, frames: StreamXout, color: bool, lossless: bool, fps: float, frame_shape: Tuple[int, ...]):
        super().__init__(frames)
        # We could use cv2.IMREAD_UNCHANGED, but it produces 3 planes (RGB) for mono frame instead of a single plane
        self.flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        self.lossless = lossless
        self.fps = fps
        self._frame_shape = frame_shape

        if lossless and self._visualizer:
            raise ValueError('Visualizing Lossless MJPEG stream is not supported!')

    def decode_frame(self, packet: FramePacket) -> np.ndarray:
        return cv2.imdecode(packet.imgFrame.getData(), self.flag)

    def visualize(self, packet: FramePacket):
        packet.frame = self.decode_frame(packet)
        super().visualize(packet)

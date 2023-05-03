from typing import Tuple

import depthai as dai

from depthai_sdk.classes import FramePacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames

try:
    import av
except ImportError:
    av = None


class XoutH26x(XoutFrames):
    def __init__(self,
                 frames: StreamXout,
                 color: bool,
                 profile: dai.VideoEncoderProperties.Profile,
                 fps: float,
                 frame_shape: Tuple[int, ...]):
        super().__init__(frames)
        self.name = 'H26x Stream'
        self.color = color
        self.profile = profile
        self.fps = fps
        self._frame_shape = frame_shape
        fourcc = 'hevc' if profile == dai.VideoEncoderProperties.Profile.H265_MAIN else 'h264'
        self.codec = av.CodecContext.create(fourcc, "r") if av else None

    def decode_frame(self, packet: FramePacket):
        if not self.codec:
            raise ImportError('av is not installed. Please install it with `pip install av`')

        enc_packets = self.codec.parse(packet.msg.getData())
        if len(enc_packets) == 0:
            return None

        frames = self.codec.decode(enc_packets[-1])
        if not frames:
            return None

        frame = frames[0].to_ndarray(format='bgr24')

        # If it's Mono, squeeze from 3 planes (height, width, 3) to single plane (height, width)
        if not self.color:
            frame = frame[:, :, 0]

        return frame

    def visualize(self, packet: FramePacket):
        decoded_frame = self.decode_frame(packet)
        if decoded_frame is None:
            return

        packet.frame = decoded_frame
        super().visualize(packet)

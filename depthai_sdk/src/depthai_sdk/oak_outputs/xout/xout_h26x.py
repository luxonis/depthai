from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.classes.packets import H26xPacket
import depthai as dai

try:
    import av
except ImportError:
    av = None


class XoutH26x(XoutFrames):
    def __init__(self,
                 frames: StreamXout,
                 is_color: bool,
                 profile: dai.VideoEncoderProperties.Profile):
        super().__init__(frames)
        self.name = 'H26x Stream'
        self.is_color = is_color
        fourcc = 'hevc' if profile == dai.VideoEncoderProperties.Profile.H265_MAIN else 'h264'
        self.codec = av.CodecContext.create(fourcc, "r") if av else None

    def new_msg(self, name: str, msg):
        if name not in self._streams:
            return
        return H26xPacket(self.get_packet_name(),
                             msg,
                             self.codec, self.is_color)


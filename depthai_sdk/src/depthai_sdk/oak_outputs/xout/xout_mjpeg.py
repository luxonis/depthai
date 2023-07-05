from depthai_sdk.classes.packets import MjpegPacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames

class XoutMjpeg(XoutFrames):
    name: str = "MJPEG Stream"

    def __init__(self,
                 frames: StreamXout,
                 is_color: bool,
                 lossless: bool):
        super().__init__(frames)
        # We could use cv2.IMREAD_UNCHANGED, but it produces 3 planes (RGB) for mono frame instead of a single plane
        self.is_color = is_color
        self.lossless = lossless

    def new_msg(self, name: str, msg) -> MjpegPacket:
        if name not in self._streams:
            return
        return MjpegPacket(self.get_packet_name(),
                             msg, self.is_color, self.lossless)

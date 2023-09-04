from typing import List, Optional

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

    def __init__(self,
                 frames: StreamXout,
                 fourcc: Optional[str] = None,  # 'mjpeg', 'h264', 'hevc'
                 ):
        """
        Args:
            frames: StreamXout object.
            fourcc: Codec to use for encoding. If None, no encoding will be done.
        """
        self.frames = frames
        self.name = frames.name
        self._fourcc = fourcc
        self._codec = None

        super().__init__()

    def set_fourcc(self, fourcc: str) -> 'XoutFrames':
        self._fourcc = fourcc
        return self

    def xstreams(self) -> List[StreamXout]:
        return [self.frames]

    def new_msg(self, name: str, msg):
        if name not in self._streams:
            return
        return FramePacket(self.get_packet_name(), msg)

    def get_codec(self):
        # No codec, frames are NV12/YUV/BGR, so we can just use imgFrame.getCvFrame()
        if self._fourcc is None:
            return None

        if self._codec is None:
            try:
                import av
            except ImportError:
                raise ImportError('Attempted to decode an encoded frame, but av is not installed.'
                                  ' Please install it with `pip install av`')
            self._codec = av.CodecContext.create(self._fourcc, "r")
        return self._codec

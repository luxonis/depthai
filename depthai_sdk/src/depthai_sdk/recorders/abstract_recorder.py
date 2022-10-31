from abc import ABC, abstractmethod
from enum import IntEnum

import depthai as dai

from depthai_sdk.oak_outputs.xout import XoutFrames, XoutH26x, XoutMjpeg, XoutDepth
from depthai_sdk.oak_outputs.xout_base import XoutBase


class Recorder(ABC):
    @abstractmethod
    def write(self, name: str, frame: dai.ImgFrame):
        pass

    @abstractmethod
    def close(self):
        pass


class OakStream:
    class StreamType(IntEnum):
        RAW = 0  # Unencoded frames (mono, color, disparity)
        MJPEG = 1
        H264 = 2
        H265 = 3
        DEPTH = 4  # 16 bit

    type: StreamType

    def __init__(self, xout: XoutBase):
        if isinstance(xout, XoutMjpeg):
            self.type = self.StreamType.MJPEG
        elif isinstance(xout, XoutH26x):
            if xout.profile == dai.VideoEncoderProperties.Profile.H265_MAIN:
                self.type = self.StreamType.H265
            else:
                self.type = self.StreamType.H264
        elif isinstance(xout, XoutDepth):
            self.type = self.StreamType.DEPTH
        elif isinstance(xout, XoutFrames):
            self.type = self.StreamType.RAW
        else:
            raise ValueError("You have passed invalid Component Output to the Recorder!")

    def fourcc(self) -> str:
        if self.type == self.StreamType.MJPEG:
            return 'mjpeg'
        elif self.type == self.StreamType.H264:
            return 'h264'
        elif self.type == self.StreamType.H265:
            return 'hevc'

    def isH265(self) -> bool:
        return self.type == self.StreamType.H265

    def isH264(self) -> bool:
        return self.type == self.StreamType.H264

    def isH26x(self) -> bool:
        return self.isH264() or self.isH265()

    def isMjpeg(self) -> bool:
        return self.type == self.StreamType.MJPEG

    def isRaw(self) -> bool:
        return self.type == self.StreamType.RAW

    def isDepth(self) -> bool:
        return self.type == self.StreamType.DEPTH

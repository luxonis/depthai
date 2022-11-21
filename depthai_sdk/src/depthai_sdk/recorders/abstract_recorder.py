from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List

import depthai as dai

from depthai_sdk.oak_outputs.xout import XoutFrames, XoutH26x, XoutMjpeg, XoutDepth, XoutDisparity, XoutIMU
from depthai_sdk.oak_outputs.xout_base import XoutBase
from pathlib import Path

class Recorder(ABC):
    @abstractmethod
    def write(self, name: str, frame: dai.ImgFrame):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, path: Path, device: dai.Device, xouts: List[XoutFrames]):
        raise NotImplementedError()

class OakStream:
    class StreamType(IntEnum):
        RAW = 0  # Unencoded frames (mono, color, disparity)
        MJPEG = 1
        H264 = 2
        H265 = 3
        DEPTH = 4  # 16 bit
        IMU = 5

    type: StreamType
    xlink_name: str

    def __init__(self, xout: XoutBase):
        if isinstance(xout, XoutMjpeg):
            self.type = self.StreamType.MJPEG
            self.xlink_name = xout.frames.name
        elif isinstance(xout, XoutH26x):
            self.xlink_name = xout.frames.name
            if xout.profile == dai.VideoEncoderProperties.Profile.H265_MAIN:
                self.type = self.StreamType.H265
            else:
                self.type = self.StreamType.H264
        elif isinstance(xout, XoutDepth):
            self.xlink_name = xout.frames.name
            self.type = self.StreamType.DEPTH
        elif isinstance(xout, XoutDisparity):
            self.xlink_name = xout.frames.name
            self.type = self.StreamType.RAW
        elif isinstance(xout, XoutFrames):
            self.xlink_name = xout.frames.name
            self.type = self.StreamType.RAW
        elif isinstance(xout, XoutIMU):
            self.xlink_name = xout.imu_out.name
            self.type = self.StreamType.IMU
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

    def isIMU(self):
        return self.type == self.StreamType.IMU

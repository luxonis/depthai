from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import List

import depthai as dai

import depthai_sdk.oak_outputs.xout as outputs


class Recorder(ABC):
    @abstractmethod
    def write(self, name: str, frame: dai.ImgFrame):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, path: Path, device: dai.Device, xouts: List['XoutFrames']):
        raise NotImplementedError()


class OakStream:
    class StreamType(IntEnum):
        RAW = 0  # Unencoded frames (mono, color, disparity)
        MJPEG = 1
        H264 = 2
        H265 = 3
        DEPTH = 4  # 16 bit
        IMU = 5

    def __init__(self, xout: outputs.xout_base.XoutBase):
        if isinstance(xout, outputs.xout_depth.XoutDisparityDepth):
            self.xlink_name = xout.frames.name
            self.type = self.StreamType.DEPTH  # TODO is depth raw or should it be DEPTH?
        elif isinstance(xout, outputs.xout_disparity.XoutDisparity) and xout._fourcc is None:
            self.xlink_name = xout.frames.name
            self.type = self.StreamType.RAW
        elif isinstance(xout, outputs.xout_frames.XoutFrames):
            self.xlink_name = xout.frames.name
            if xout._fourcc is None:
                self.type = self.StreamType.RAW
            elif xout._fourcc == 'hevc':
                self.type = self.StreamType.H265
            elif xout._fourcc == 'h264':
                self.type = self.StreamType.H264
            elif xout._fourcc == 'mjpeg':
                self.type = self.StreamType.MJPEG

        elif isinstance(xout, outputs.xout_imu.XoutIMU):
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
        elif self.type == self.StreamType.DEPTH:
            return 'y16'

    def is_h265(self) -> bool:
        return self.type == self.StreamType.H265

    def is_h264(self) -> bool:
        return self.type == self.StreamType.H264

    def is_h26x(self) -> bool:
        return self.is_h264() or self.is_h265()

    def is_mjpeg(self) -> bool:
        return self.type == self.StreamType.MJPEG

    def is_raw(self) -> bool:
        return self.type == self.StreamType.RAW

    def is_depth(self) -> bool:
        return self.type == self.StreamType.DEPTH

    def is_imu(self):
        return self.type == self.StreamType.IMU

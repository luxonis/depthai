import logging
from pathlib import Path
from typing import Dict

import depthai as dai
from mcap_ros1.writer import Writer as Ros1Writer

from depthai_sdk.integrations.ros.ros_base import RosBase
# from depthai_sdk.integrations.ros.depthai2ros import DepthAi2Ros1
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.recorders.abstract_recorder import *


class McapRecorder(Recorder, RosBase):
    """
    This is a helper class that lets you save frames into mcap (.mcap), which can be replayed using Foxglove studio app.
    """

    def __init__(self):
        self.path = None
        self.converter = None
        self.stream = None
        self.ros_writer = None

        self._closed = False
        self._pcl = False
        self._stream_type: Dict[str, OakStream] = dict()
        self._name_mapping: Dict[str, str] = dict()  # XLink name to nice name mapping

    def update(self, path: Path, device: dai.Device, xouts: List['XoutFrames']):
        """
        Args:
            path (Path): Path to which we record
            device (dai.Device): OAK Device
            xouts (List['XoutFrames']): List of outputs, which are used to record
        """
        self.path = str(path / "recordings.mcap")
        self.converter = Bridge(device)
        self.stream = open(self.path, "w+b")
        self.ros_writer = Ros1Writer(output=self.stream)

        RosBase.__init__(self)
        RosBase.update(self, device, xouts)

        for xout in xouts:
            if xout.is_h26x():
                raise Exception("MCAP recording only supports MJPEG encoding!")
            if xout.is_mjpeg() and xout.lossless:
                # Foxglove Studio doesn't (yet?) support Lossless MJPEG
                raise Exception("MCAP recording doesn't support Lossless MJPEG encoding!")

    def write(self, name: str, frame: dai.ImgFrame):
        self.new_msg(name, frame)  # To RosMsg which arrives to new_ros_msg()

    def new_ros_msg(self, topic: str, ros_msg):
        self.ros_writer.write_message(topic, ros_msg)

    def set_pointcloud(self, enable: bool):
        """
        Whether to convert depth to pointcloud
        """
        self._pcl = enable

    # def write(self, name: str, frame: dai.ImgFrame):
    #     if name not in self._name_mapping:
    #         return
    #
    #     name = self._name_mapping[name]
    #     if self._stream_type[name].is_depth() and self._pcl:
    #         msg = self.converter.PointCloud2(frame)
    #         self.ros_writer.write_message(f"pointcloud/raw", msg)
    #         # tf = self.converter.TfMessage(frame)
    #         # self.ros_writer.write_message(f"pointcloud/tf", tf)
    #     elif self._stream_type[name].is_mjpeg():
    #         msg = self.converter.CompressedImage(frame)
    #         self.ros_writer.write_message(f"{name}/compressed", msg)
    #     elif self._stream_type[name].is_imu():
    #         frame: dai.IMUData
    #         for imu_packet in frame.packets:
    #             msg = self.converter.Imu(imu_packet)
    #             self.ros_writer.write_message(f"imu", msg)
    #     else:  # Non-encoded frame; rgb/mono/depth
    #         msg = self.converter.Image(frame)
    #         self.ros_writer.write_message(f"{name}/raw", msg)

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        self.ros_writer.finish()
        self.stream.close()
        logging.info(f'.MCAP recording saved at {self.path}')

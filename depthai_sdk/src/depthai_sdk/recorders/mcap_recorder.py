from pathlib import Path
from typing import Dict
import depthai as dai
from mcap_ros1.writer import Writer as Ros1Writer
from depthai_sdk.recorders.abstract_recorder import *
from depthai_sdk.integrations.ros.depthai2ros import Bridge
from depthai_sdk.oak_outputs.xout import XoutFrames
from depthai_sdk.integrations.ros.ros_base import RosBase

class McapRecorder(Recorder, RosBase):
    '''
    This is a helper class that lets you save frames into mcap (.mcap), which can be replayed using Foxglove studio app.
    '''

    _closed = False

    def update(self, path: Path, device: dai.Device, xouts: List[XoutFrames]):
        """
        Args:
            path (Path): Path to which we record
            device (dai.Device): OAK Device
        """
        self.path = str(path / "recordings.mcap")
        self.converter = Bridge(device)
        self.stream = open(self.path, "w+b")
        self.ros_writer = Ros1Writer(output=self.stream)

        RosBase.__init__(self)
        RosBase.update(self, device, xouts)

        for xout in xouts:
            if xout.isH26x():
                raise Exception("MCAP recording only supports MJPEG encoding!")
            if xout.isMjpeg() and xout.lossless:
                # Foxglove Studio doesn't (yet?) support Lossless MJPEG
                raise Exception("MCAP recording doesn't support Lossless MJPEG encoding!")
            # rec.setPointcloud(self._pointcloud)

    def write(self, name: str, frame: dai.ImgFrame):
        self.new_msg(name, frame)  # To RosMsg which arrives to new_ros_msg()

    def new_ros_msg(self, topic: str, ros_msg):
        self.ros_writer.write_message(topic, ros_msg)

    def setPointcloud(self, enable: bool):
        """
        Whether to convert depth to pointcloud
        """
        self.pointcloud = enable

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        self.ros_writer.finish()
        self.stream.close()
        print(".MCAP recording saved at", self.path)

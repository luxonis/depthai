'''
This is a helper class that let's you save frames into mcap (.mcap), which can be replayed using Foxglove studio app.
'''

import numpy as np
from pathlib import Path

from mcap_ros1.writer import Writer as Ros1Writer
from .abstract_recorder import Recorder
from .depthai2ros import DepthAi2Ros1

import depthai as dai


class McapRecorder(Recorder):
    _closed = False
    _pcl = False
    def __init__(self, path: Path, device: dai.Device):
        """
        
        Args:
            path (Path): Path to which we record
            device (dai.Device): OAK Device
        """
        self.converter = DepthAi2Ros1(device)
        self.path = str(path / "recordings.mcap")
        self.stream = open(self.path, "w+b")
        self.ros_writer = Ros1Writer(output=self.stream)
    
    def setPointcloud(self, enable: bool):
        """
        Whether to convert depth to pointcloud
        """
        self._pcl = enable

    def write(self, name: str, frame: dai.ImgFrame):
        if name == "depth":
            if self._pcl: # Generate pointcloud from depth and save it
                msg = self.converter.PointCloud2(frame)
                self.ros_writer.write_message(f"pointcloud/raw", msg)
            else: # Save raw depth frame
                msg = self.converter.Image(frame)
                self.ros_writer.write_message(f"depth/raw", msg)
        else:
            msg = self.converter.CompressedImage(frame)
            self.ros_writer.write_message(f"{name}/compressed", msg)
        
    def close(self) -> None:
        if self._closed: return
        self._closed = True
        self.ros_writer.finish()
        self.stream.close()
        print(".MCAP recording saved at", self.path)

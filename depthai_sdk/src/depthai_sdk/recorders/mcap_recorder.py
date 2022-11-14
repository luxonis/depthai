from pathlib import Path
from typing import List, Dict

import depthai as dai
from mcap_ros1.writer import Writer as Ros1Writer
from depthai_sdk.recorders.abstract_recorder import *
from depthai_sdk.recorders.depthai2ros import DepthAi2Ros1


class McapRecorder(Recorder):
    '''
    This is a helper class that lets you save frames into mcap (.mcap), which can be replayed using Foxglove studio app.
    '''

    _closed = False
    _pcl = False
    _stream_type: Dict[str, OakStream]

    def update(self, path: Path, device: dai.Device, xouts: List[XoutFrames]):
        """
        Args:
            path (Path): Path to which we record
            device (dai.Device): OAK Device
        """
        self.path = str(path / "recordings.mcap")
        self.converter = DepthAi2Ros1(device)
        self.stream = open(self.path, "w+b")
        self.ros_writer = Ros1Writer(output=self.stream)

        self._stream_type = dict()
        for xout in xouts:
            name = xout.frames.friendly_name or xout.frames.name
            codec = OakStream(xout)
            self._stream_type[name] = codec

            if codec.isH26x():
                raise Exception("MCAP recording only supports MJPEG encoding!")
            if codec.isMjpeg() and xout.lossless:
                # Foxglove Studio doesn't (yet?) support Lossless MJPEG
                raise Exception("MCAP recording doesn't support Lossless MJPEG encoding!")
            # rec.setPointcloud(self._pointcloud)

    def setPointcloud(self, enable: bool):
        """
        Whether to convert depth to pointcloud
        """
        self._pcl = enable

    def write(self, name: str, frame: dai.ImgFrame):
        if self._stream_type[name].isDepth():
            if self._pcl:  # Generate pointcloud from depth and save it
                msg = self.converter.PointCloud2(frame)
                self.ros_writer.write_message(f"pointcloud/raw", msg)
            else:  # Save raw depth frame
                msg = self.converter.Image(frame)
                self.ros_writer.write_message(f"depth/raw", msg)
        elif self._stream_type[name].isMjpeg():
            msg = self.converter.CompressedImage(frame)
            self.ros_writer.write_message(f"{name}/compressed", msg)

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        self.ros_writer.finish()
        self.stream.close()
        print(".MCAP recording saved at", self.path)

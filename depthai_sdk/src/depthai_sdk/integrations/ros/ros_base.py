from typing import Any, List, Dict

import depthai as dai
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, Imu

from depthai_sdk.integrations.ros.depthai2ros2 import DepthAi2Ros2
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames


class RosStream:
    datatype: dai.DatatypeEnum  # dai datatype of the stream
    xout: XoutBase  # From which Xout is the stream
    topic: str  # Topic name, eg. color/compressed
    ros_type: Any  # CompressedImage, Image, IMU, PointCloud2...


class RosBase:
    """
    Base class that is used by ros streaming component and mcap recorder.
    """

    def __init__(self):
        self.streams: Dict[str, RosStream]  # key = xlink stream name
        self.pointcloud: bool = False  # Convert depth -> pointcloud
        self.bridge: DepthAi2Ros2
        self.streams = dict()

    def update(self, device: dai.Device, xouts: List[XoutFrames]):
        self.bridge = DepthAi2Ros2(device)

        for xout in xouts:
            for stream in xout.xstreams():
                rs = RosStream()
                rs.datatype = stream.stream.possibleDatatypes[0].datatype
                name = xout.name.lower()

                if xout.is_depth() and not stream.name.endswith('depth'):
                    # Mono right frame for WLS, skip
                    continue

                if xout.is_mjpeg():
                    rs.topic = f'/{name}/compressed'
                    rs.ros_type = CompressedImage
                elif xout.is_depth() and self.pointcloud:
                    rs.topic = '/pointcloud/raw'
                    rs.ros_type = PointCloud2
                elif xout.is_imu():
                    rs.topic = '/imu'
                    rs.ros_type = Imu
                else:  # Non-encoded frames; rgb, mono, depth
                    rs.topic = f'/{name}/raw'
                    rs.ros_type = Image

                self.streams[stream.name] = rs

    def new_ros_msg(self, topic: str, ros_msg) -> None:
        raise NotImplementedError('Abstract function, override it!')

    def new_msg(self, name: str, dai_msg: dai.ADatatype):
        if name not in self.streams:  # Not relevant
            return

        # depthai msgs, name = xlink name
        stream = self.streams[name]

        msg = None
        if stream.ros_type == CompressedImage:
            dai_msg: dai.ImgFrame
            msg = self.bridge.CompressedImage(dai_msg)
        # elif stream.ros_type == PointCloud2:
        #     msg = self.bridge.PointCloud2(dai_msg)
        elif stream.ros_type == Imu:
            msg = self.bridge.Imu(dai_msg)
        elif stream.ros_type == Image:
            dai_msg: dai.ImgFrame
            msg = self.bridge.Image(dai_msg)

        self.new_ros_msg(stream.topic, msg)

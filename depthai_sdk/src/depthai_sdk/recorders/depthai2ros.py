import depthai as dai

from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField #s, PointCloud
# from geometry_msgs.msg import Vector3, Quaternion, Pose2D, Point
# from std_msgs.msg import Header, ColorRGBA, String
# from visualization_msgs.msg import ImageMarker
# from geometry_msgs.msg import Vector3, Quaternion, Pose2D, Point
from genpy.rostime import Time
import numpy as np
import time

class DepthAi2Ros1:
    xyz = dict()

    def __init__(self, device: dai.Device) -> None:
        self.start_nanos = time.time_ns()
        self.device = device

    def getTime(self) -> Time:
        ts = time.time_ns() - self.start_nanos
        return Time(int(ts/1000000), ts%1000000)

    def CompressedImage(self, imgFrame: dai.ImgFrame) -> CompressedImage:
        msg = CompressedImage()
        msg.header.stamp = self.getTime()
        msg.format = "jpeg"
        msg.data = np.array(imgFrame.getData()).tobytes()
        return msg

    def Image(self, imgFrame: dai.ImgFrame) -> Image:
        msg = Image()
        # print(imgFrame.getType()) # Check whether this is RGB888p frame
        # dai.ImgFrame.Type.RAW16 == depth
        msg.header.stamp = self.getTime()
        msg.height = imgFrame.getHeight()
        msg.width = imgFrame.getWidth()
        msg.encoding = 'mono16' # if rgb else 'mono16', # For depth
        msg.is_bigendian = 0
        msg.step = imgFrame.getWidth() * 2 # *2 for mono16 (depth)
        msg.data = np.array(imgFrame.getData()).tobytes()
        return msg


    def PointCloud2(self, imgFrame: dai.ImgFrame):
        """
        Depth frame -> ROS1 PointCloud2 message
        """
        msg = PointCloud2()
        msg.header.stamp = self.getTime()

        heigth = str(imgFrame.getHeight())
        if heigth not in self.xyz:
            self._create_xyz(imgFrame.getWidth(), imgFrame.getHeight())
        
        frame = imgFrame.getCvFrame()
        frame = np.expand_dims(np.array(frame), axis=-1)
        pcl = self.xyz[heigth] * frame / 1000.0 # To meters

        msg.height = imgFrame.getHeight()
        msg.width = imgFrame.getWidth()
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12 # 3 * float32 (=4 bytes)
        msg.row_step = 12 * imgFrame.getWidth()
        msg.data = pcl.tobytes()
        msg.is_dense = True
        return msg

    def _create_xyz(self, width, height):
        calibData = self.device.readCalibration()
        M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(width, height))
        camera_matrix = np.array(M_right).reshape(3, 3)

        xs = np.linspace(0, width - 1, width, dtype=np.float32)
        ys = np.linspace(0, height - 1, height, dtype=np.float32)

        # generate grid by stacking coordinates
        base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
        points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2

        # unpack coordinates
        u_coord: np.array = points_2d[..., 0]
        v_coord: np.array = points_2d[..., 1]

        # unpack intrinsics
        fx: np.array = camera_matrix[0, 0]
        fy: np.array = camera_matrix[1, 1]
        cx: np.array = camera_matrix[0, 2]
        cy: np.array = camera_matrix[1, 2]

        # projective
        x_coord: np.array = (u_coord - cx) / fx
        y_coord: np.array = (v_coord - cy) / fy

        xyz = np.stack([x_coord, y_coord], axis=-1)
        self.xyz[str(height)] = np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)


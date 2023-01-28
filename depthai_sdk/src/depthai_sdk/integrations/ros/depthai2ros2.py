import time
from typing import Tuple, List, Union
import depthai as dai
import numpy as np
import rclpy
import rclpy.node as node

# from geometry_msgs.msg import Vector3, Quaternion, Pose2D, Point, Transform, TransformStamped
# from std_msgs.msg import Header, ColorRGBA, String
# from visualization_msgs.msg import ImageMarker
from geometry_msgs.msg import Vector3, Quaternion

from sensor_msgs.msg import CompressedImage, Image, Imu  # , PointCloud2, PointField, Imu  # s, PointCloud
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from depthai_sdk.integrations.ros.imu_interpolation import ImuInterpolation


class DepthAi2Ros2:
    xyz = dict()

    def __init__(self, device: dai.Device) -> None:
        self.start_time = dai.Clock.now()
        self.device = device
        self.imu_packets = []
        self.imu_interpolation = ImuInterpolation()

    def set_header(self, msg, dai_msg: Union[dai.ImgFrame, dai.IMUReport]) -> Header:
        try:
            msg.header.frame_id = str(dai_msg.getSequenceNum())  # ImgFrame
        except:
            msg.header.frame_id = str(dai_msg.sequence)  # IMUReport

        ts = dai_msg.getTimestampDevice() - self.start_time
        # secs / nanosecs
        msg.header.stamp = Time(sec=ts.seconds, nanosec=ts.microseconds * 1000)
        return msg

    def CompressedImage(self, imgFrame: dai.ImgFrame) -> CompressedImage:
        msg = CompressedImage()
        self.set_header(msg, imgFrame)
        msg.format = "jpeg"
        msg.data.frombytes(imgFrame.getData())
        return msg

    def Image(self, imgFrame: dai.ImgFrame) -> Image:
        msg = Image()
        self.set_header(msg, imgFrame)
        msg.height = imgFrame.getHeight()
        msg.width = imgFrame.getWidth()
        msg.step = imgFrame.getWidth()
        msg.is_bigendian = 0

        type = imgFrame.getType()
        TYPE = dai.ImgFrame.Type
        if type == TYPE.RAW16:  # Depth
            msg.encoding = 'mono16'
            msg.step *= 2  # 2 bytes per pixel
            msg.data.frombytes(imgFrame.getData())
        elif type in [TYPE.GRAY8, TYPE.RAW8]:  # Mono frame
            msg.encoding = 'mono8'
            msg.data.frombytes(imgFrame.getData())
        else:
            msg.encoding = 'bgr8'
            msg.data.frombytes(imgFrame.getCvFrame())
        return msg

    # def TfMessage(self,
    #               imgFrame: dai.ImgFrame,
    #               translation: Tuple[float, float, float] = (0., 0., 0.),
    #               rotation: Tuple[float, float, float, float] = (0., 0., 0., 0.)) -> tfMessage:
    #     msg = tfMessage()
    #     tf = TransformStamped()
    #     tf.header = self.header(imgFrame)
    #     tf.child_frame_id = str(imgFrame.getSequenceNum())
    #     tf.transform = Transform(
    #         translation=Vector3(x=translation[0], y=translation[1], z=translation[2]),
    #         rotation=Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])
    #     )
    #     msg.transforms.append(tf)
    #     return msg
    #
    # def PointCloud2(self, imgFrame: dai.ImgFrame) -> PointCloud2:
    #     """
    #     Depth frame -> ROS1 PointCloud2 message
    #     """
    #     msg = PointCloud2()
    #     msg.header = self.header(imgFrame)
    #
    #     heigth = str(imgFrame.getHeight())
    #     if heigth not in self.xyz:
    #         self._create_xyz(imgFrame.getWidth(), imgFrame.getHeight())
    #
    #     frame = imgFrame.getCvFrame()
    #     frame = np.expand_dims(np.array(frame), axis=-1)
    #     pcl = self.xyz[heigth] * frame / 1000.0  # To meters
    #
    #     msg.height = imgFrame.getHeight()
    #     msg.width = imgFrame.getWidth()
    #     msg.fields = [
    #         PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    #         PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    #         PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    #     ]
    #     msg.is_bigendian = False
    #     msg.point_step = 12  # 3 * float32 (=4 bytes)
    #     msg.row_step = 12 * imgFrame.getWidth()
    #     msg.data = pcl.tobytes()
    #     msg.is_dense = True
    #     return msg
    #
    # def _create_xyz(self, width, height):
    #     calibData = self.device.readCalibration()
    #     M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(width, height))
    #     camera_matrix = np.array(M_right).reshape(3, 3)
    #
    #     xs = np.linspace(0, width - 1, width, dtype=np.float32)
    #     ys = np.linspace(0, height - 1, height, dtype=np.float32)
    #
    #     # generate grid by stacking coordinates
    #     base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
    #     points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2
    #
    #     # unpack coordinates
    #     u_coord: np.array = points_2d[..., 0]
    #     v_coord: np.array = points_2d[..., 1]
    #
    #     # unpack intrinsics
    #     fx: np.array = camera_matrix[0, 0]
    #     fy: np.array = camera_matrix[1, 1]
    #     cx: np.array = camera_matrix[0, 2]
    #     cy: np.array = camera_matrix[1, 2]
    #
    #     # projective
    #     x_coord: np.array = (u_coord - cx) / fx
    #     y_coord: np.array = (v_coord - cy) / fy
    #
    #     xyz = np.stack([x_coord, y_coord], axis=-1)
    #     self.xyz[str(height)] = np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
    def Imu(self, dai_msg):
        dai_msg: dai.IMUData
        for packet in dai_msg.packets:
            msg = Imu(
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vector3(x=0.0,y=0.0, z=0.0),
                linear_acceleration_covariance=np.zeros(9)
            )
            report = packet.acceleroMeter or packet.gyroscope or packet.magneticField or packet.rotationVector
            self.set_header(msg, report)
            self.imu_interpolation.Imu(msg, packet)
            # TODO: publish from here directly, so single IMUData can result in more Imu packets?
            return msg


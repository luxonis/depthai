#
# from typing import Tuple
#
# import depthai as dai
# import numpy as np
# import std_msgs
# from genpy.rostime import Time
# # from std_msgs.msg import Header, ColorRGBA, String
# # from visualization_msgs.msg import ImageMarker
# from geometry_msgs.msg import TransformStamped
# from geometry_msgs.msg import Vector3, Quaternion, Transform
# from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField, Imu  # s, PointCloud
# from tf.msg import tfMessage
# from .imu_interpolation import ImuSyncMethod
#
# """
# --extra-index-url https://rospypi.github.io/simple/
# sensor_msgs
# geometry_msgs
# std_msgs
# genpy
# tf2_msgs
# tf2_ros
# tf
# """
#
#
# class DepthAi2Ros1:
#     def __init__(self, device: dai.Device) -> None:
#         self.start_time = dai.Clock.now()
#         self.device = device
#
#         self.xyz = dict()
#
#     def header(self, msg: dai.Buffer) -> std_msgs.msg.Header:
#         header = std_msgs.msg.Header()
#         ts = (msg.getTimestamp() - self.start_time).total_seconds()
#         # secs / nanosecs
#         header.stamp = Time(int(ts), (ts % 1) * 1e6)
#         try:
#             header.frame_id = str(msg.getSequenceNum())  # ImgFrame
#         except:
#             header.frame_id = str(msg.sequence)  # IMUReport
#         return header
#
#     def CompressedImage(self, imgFrame: dai.ImgFrame) -> CompressedImage:
#         msg = CompressedImage()
#         msg.header = self.header(imgFrame)
#         msg.format = "jpeg"
#         msg.data = imgFrame.getData().tobytes()
#         return msg
#
#     def Imu(self, imu_packet: dai.IMUPacket, sync_mode: ImuSyncMethod = ImuSyncMethod.LINEAR_INTERPOLATE_ACCEL,
#             linear_accel_cov: float = 0., angular_velocity_cov: float = 0) -> Imu:
#         if len(self.imu_packets) > 20:
#             self.imu_packets.pop(0)
#
#         self.imu_packets.append(imu_packet)
#
#         if sync_mode != ImuSyncMethod.COPY:
#             interp_imu_packets = self.fillImuData_LinearInterpolation(self, sync_mode)
#             if len(interp_imu_packets) > 0:
#                 imu_packet = interp_imu_packets[-1]
#
#         msg = Imu()
#
#         if imu_packet.acceleroMeter is not None:
#             msg.header = self.header(imu_packet.acceleroMeter)
#             msg.linear_acceleration.x = imu_packet.acceleroMeter.x
#             msg.linear_acceleration.y = imu_packet.acceleroMeter.y
#             msg.linear_acceleration.z = imu_packet.acceleroMeter.z
#
#         if imu_packet.gyroscope is not None:
#             msg.header = self.header(imu_packet.gyroscope)
#             msg.angular_velocity.x = imu_packet.gyroscope.x
#             msg.angular_velocity.y = imu_packet.gyroscope.y
#             msg.angular_velocity.z = imu_packet.gyroscope.z
#
#         msg.orientation.x = 0.0
#         msg.orientation.y = 0.0
#         msg.orientation.z = 0.0
#         msg.orientation.w = 1.0
#
#         msg.orientation_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#         msg.linear_acceleration_covariance = [linear_accel_cov, 0.0, 0.0, 0.0, linear_accel_cov, 0.0, 0.0, 0.0,
#                                               linear_accel_cov]
#         msg.angular_velocity_covariance = [angular_velocity_cov, 0.0, 0.0, 0.0, angular_velocity_cov, 0.0, 0.0, 0.0,
#                                            angular_velocity_cov]
#
#         return msg
#
#
#
#     def Image(self, imgFrame: dai.ImgFrame) -> Image:
#         msg = Image()
#         msg.header = self.header(imgFrame)
#         msg.height = imgFrame.getHeight()
#         msg.width = imgFrame.getWidth()
#         msg.step = imgFrame.getWidth()
#         msg.is_bigendian = 0
#
#         type = imgFrame.getType()
#         TYPE = dai.ImgFrame.Type
#         if type == TYPE.RAW16:  # Depth
#             msg.encoding = 'mono16'
#             msg.step *= 2  # 2 bytes per pixel
#             msg.data = imgFrame.getData().tobytes()
#         elif type in [TYPE.GRAY8, TYPE.RAW8]:  # Mono frame
#             msg.encoding = 'mono8'
#             msg.data = imgFrame.getData().tobytes()  # np.array(imgFrame.getFrame()).tobytes()
#         else:
#             msg.encoding = 'bgr8'
#             msg.data = imgFrame.getCvFrame().tobytes()
#         return msg
#
#     def TfMessage(self,
#                   imgFrame: dai.ImgFrame,
#                   translation: Tuple[float, float, float] = (0., 0., 0.),
#                   rotation: Tuple[float, float, float, float] = (0., 0., 0., 0.)) -> tfMessage:
#         msg = tfMessage()
#         tf = TransformStamped()
#         tf.header = self.header(imgFrame)
#         tf.child_frame_id = str(imgFrame.getSequenceNum())
#         tf.transform = Transform(
#             translation=Vector3(x=translation[0], y=translation[1], z=translation[2]),
#             rotation=Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])
#         )
#         msg.transforms.append(tf)
#         return msg
#
#     def PointCloud2(self, imgFrame: dai.ImgFrame) -> PointCloud2:
#         """
#         Depth frame -> ROS1 PointCloud2 message
#         """
#         msg = PointCloud2()
#         msg.header = self.header(imgFrame)
#
#         heigth = str(imgFrame.getHeight())
#         if heigth not in self.xyz:
#             self._create_xyz(imgFrame.getWidth(), imgFrame.getHeight())
#
#         frame = imgFrame.getCvFrame()
#         frame = np.expand_dims(np.array(frame), axis=-1)
#         pcl = self.xyz[heigth] * frame / 1000.0  # To meters
#
#         msg.height = imgFrame.getHeight()
#         msg.width = imgFrame.getWidth()
#         msg.fields = [
#             PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
#             PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
#             PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
#         ]
#         msg.is_bigendian = False
#         msg.point_step = 12  # 3 * float32 (=4 bytes)
#         msg.row_step = 12 * imgFrame.getWidth()
#         msg.data = pcl.tobytes()
#         msg.is_dense = True
#         return msg
#
#     def _create_xyz(self, width, height):
#         calibData = self.device.readCalibration()
#         M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(width, height))
#         camera_matrix = np.array(M_right).reshape(3, 3)
#
#         xs = np.linspace(0, width - 1, width, dtype=np.float32)
#         ys = np.linspace(0, height - 1, height, dtype=np.float32)
#
#         # generate grid by stacking coordinates
#         base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
#         points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2
#
#         # unpack coordinates
#         u_coord: np.array = points_2d[..., 0]
#         v_coord: np.array = points_2d[..., 1]
#
#         # unpack intrinsics
#         fx: np.array = camera_matrix[0, 0]
#         fy: np.array = camera_matrix[1, 1]
#         cx: np.array = camera_matrix[0, 2]
#         cy: np.array = camera_matrix[1, 2]
#
#         # projective
#         x_coord: np.array = (u_coord - cx) / fx
#         y_coord: np.array = (v_coord - cy) / fy
#
#         xyz = np.stack([x_coord, y_coord], axis=-1)
#         self.xyz[str(height)] = np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)

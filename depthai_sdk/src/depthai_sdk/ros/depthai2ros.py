import depthai as dai

from sensor_msgs.msg import CompressedImage, CameraInfo, Image, Imu
from geometry_msgs.msg import Vector3, Quaternion, Pose2D, Point
from std_msgs.msg import Header, ColorRGBA, String
from visualization_msgs.msg import ImageMarker

import rospy
import numpy as np
from mcap_ros1.writer import Writer as Ros1Writer

class DepthAi2Ros1:
    # def __init__(self) -> None:

        # output = open("example.mcap", "w+b")
        # ros_writer = Ros1Writer(output=output)
        # ros_writer.write_message("chatter", String(data=f"string message {i}"))

    def CompressedImage(self, imgFrame: dai.ImgFrame) -> CompressedImage:
        msg = CompressedImage()
        print(imgFrame.getType()) # Check whether this is MJPEG encoded frame
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(imgFrame.getData()).tobytes()
        return msg

    def Image(self, imgFrame: dai.ImgFrame) -> Image:
        msg = Image()
        # print(imgFrame.getType()) # Check whether this is RGB888p frame
        msg.header.stamp = rospy.Time.now()
        msg.height = imgFrame.getHeight()
        msg.width = imgFrame.getWidth()
        msg.encoding = 'rgb8' # if rgb else 'mono16', # For depth
        msg.is_bigendian = 0
        msg.step = imgFrame.getWidth() # *2 if mono16 (depth)
        msg.data = np.array(imgFrame.getData()).tobytes()
        return msg

    # def Imu(self, imuPacket: dai.IMUPacket) -> Imu:
    #     msg = Imu()
    #     msg.accelerometer=Vector3(x=accel.x, y=accel.y, z=accel.z),
    #     msg.accelerometer_metadata=self.create_imu_metadata(accel),
    #     msg.gyroscope=Vector3(x=gyro.x, y=gyro.y, z=gyro.z),
    #     msg.gyroscope_metadata=self.create_imu_metadata(gyro),
    #     msg.magnetometer=Vector3(x=mag.x, y=mag.y, z=mag.z),
    #     msg.magnetometer_metadata=self.create_imu_metadata(mag),
    #     msg.rotation_vector=Quaternion(x=rotation.i, y=rotation.j, z=rotation.k, w=rotation.real),
    #     msg.rotation_vector_accuracy=rotation.rotationVectorAccuracy,
    #     msg.rotation_vector_metadata=self.create_imu_metadata(rotation),


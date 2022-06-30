import depthai as dai

from sensor_msgs.msg import CompressedImage, Image
# from geometry_msgs.msg import Vector3, Quaternion, Pose2D, Point
# from std_msgs.msg import Header, ColorRGBA, String
# from visualization_msgs.msg import ImageMarker
# from geometry_msgs.msg import Vector3, Quaternion, Pose2D, Point
from genpy.rostime import Time
import rospy
import numpy as np
import time

class DepthAi2Ros1:
    def __init__(self) -> None:
        self.start_nanos = time.time_ns()
        # output = open("example.mcap", "w+b")
        # ros_writer = Ros1Writer(output=output)
        # ros_writer.write_message("chatter", String(data=f"string message {i}"))
    def getTime(self) -> Time:
        ts = time.time_ns() - self.start_nanos
        return Time(int(ts/1000000), ts%1000000)

    def CompressedImage(self, imgFrame: dai.ImgFrame) -> CompressedImage:
        msg = CompressedImage()
        # print(imgFrame.getType()) # Check whether this is MJPEG encoded frame
        msg.header.stamp = self.getTime() #rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(imgFrame.getData()).tobytes()
        return msg

    def Image(self, imgFrame: dai.ImgFrame) -> Image:
        msg = Image()
        # print(imgFrame.getType()) # Check whether this is RGB888p frame
        msg.header.stamp = self.getTime() #rospy.Time.now()
        msg.height = imgFrame.getHeight()
        msg.width = imgFrame.getWidth()
        msg.encoding = 'rgb8' # if rgb else 'mono16', # For depth
        msg.is_bigendian = 0
        msg.step = imgFrame.getWidth() # *2 if mono16 (depth)
        msg.data = np.array(imgFrame.getData()).tobytes()
        return msg

    



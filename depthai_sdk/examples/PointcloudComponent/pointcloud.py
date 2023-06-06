import cv2
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import PointcloudPacket, FramePacket
import rerun as rr
import subprocess
import time

subprocess.Popen(["rerun", "--memory-limit", "200MB"])
time.sleep(1)  # Wait til rerun spins up
rr.init("Rerun ", spawn=False)
rr.connect()

def callback(packet: PointcloudPacket):
    print(
        '+'
    )
    colors = packet.color_frame.getCvFrame()[..., ::-1] # BGR to RGB
    rr.log_image('Color Image', colors)
    points = packet.points.reshape(-1, 3)
    rr.log_points("Pointcloud", points, colors=colors.reshape(-1, 3))


with OakCamera() as oak:
    pcl = oak.create_pointcloud()
    oak.callback(pcl, callback=callback)
    oak.start(blocking=True)

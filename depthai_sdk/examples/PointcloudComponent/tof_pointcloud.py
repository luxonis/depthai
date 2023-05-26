from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import PointcloudPacket, FramePacket
import rerun as rr # pip install rerun-sdk
import subprocess

# Requires custom depthai library (develop)
# pip install depthai==depthai==2.21.2.0.dev0+72cddbb885eb6495ce74ffbe80e1492c612a5452

def callback(packet: PointcloudPacket):
    points = packet.points.reshape(-1, 3)
    rr.log_points("Pointcloud", points)

with OakCamera() as oak:
    subprocess.Popen(["rerun", "--memory-limit", "200MB"])
    rr.init("Rerun ", spawn=False)
    rr.connect()
    tof = oak.create_tof()
    pcl = oak.create_pointcloud(input=tof)
    oak.callback(pcl, callback=callback)
    oak.start(blocking=True)
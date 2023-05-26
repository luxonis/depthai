from black.brackets import Depth
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import PointcloudPacket, FramePacket, DepthPacket
# import rerun as rr # pip install rerun-sdk
import depthai_viewer as viewer
import depthai as dai
import numpy as np
viewer.init("Depthai Viewer")
viewer.connect()

# Requires custom depthai library (develop)
# pip install depthai==2.21.2.0.dev0+72cddbb885eb6495ce74ffbe80e1492c612a5452 --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/

intrinsics = None
with OakCamera() as oak:
    tof = oak.create_tof()

    calib = oak.device.readCalibration()
    def callback(packet: DepthPacket):
        global intrinsics
        depth_frame: dai.ImgFrame = packet.msg
        width = depth_frame.getWidth()
        height = depth_frame.getHeight()

        if intrinsics is None:
            calibData = oak.device.readCalibration()
            intrinsics = calibData.getCameraIntrinsics(tof.camera_socket, dai.Size2f(width, height))
            intrinsics = np.array(intrinsics).reshape(3, 3)

        viewer.log_rigid3(f"world", child_from_parent=([0, 0, 0], [1,0,0,0]), xyz="RDF")
        viewer.log_pinhole("world/camera",
                        child_from_parent = intrinsics,
                        width = width,
                        height = height)
        viewer.log_depth_image("world/camera/depth", depth_frame.getFrame())

    oak.callback(tof.out.depth, callback=callback)
    oak.visualize(tof.out.amplitude)
    oak.visualize(tof.out.depth)
    oak.start(blocking=True)
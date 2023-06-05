from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import PointcloudPacket, FramePacket
import rerun as rr
import subprocess
import depthai as dai
import cv2

subprocess.Popen(["rerun", "--memory-limit", "200MB"])
rr.init("Rerun ", spawn=False)
rr.connect()

def callback(packet: PointcloudPacket):
    colors = packet.color_frame.getCvFrame()[..., ::-1] # BGR to RGB
    colors = cv2.pyrDown(colors) # 2x downscale
    rr.log_image('Color Image', colors)
    points = packet.points.reshape(-1, 3)
    print('len color', colors.shape, 'points shape', points.shape)
    rr.log_points("Pointcloud", points, colors=colors.reshape(-1, 3))

with OakCamera() as oak:
    left = oak.create_camera('left')
    right = oak.create_camera('right')

    stereo = oak.create_stereo(left=left, right=right)
    config = stereo.node.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 100 # 10cm
    config.postProcessing.thresholdFilter.maxRange = 2000 # 2m
    config.postProcessing.decimationFilter.decimationFactor = 2
    config.postProcessing.decimationFilter.decimationMode = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
    stereo.node.initialConfig.set(config)

    pcl = oak.create_pointcloud(stereo=stereo, colorize=right)
    oak.callback(pcl, callback=callback)
    oak.start(blocking=True)
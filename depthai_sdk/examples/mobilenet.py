import cv2
import depthai as dai
from depthai_sdk import Camera, AspectRatioResizeMode

# with Camera(recording='cars-california-01') as cam:
with Camera() as cam:
    color = cam.create_camera('color', out='color')
    nn = cam.create_nn('mobilenet-ssd', color, out='dets')
    nn.configNn(passthroughOut=True)
    nn.setAspectRatioResizeMode(AspectRatioResizeMode.LETTERBOX)

    # cam.callback([color, nn], test)
    cam.create_visualizer([color, nn])  # 1080P -> 720P
    cam.start(blocking=True, visualize=True)
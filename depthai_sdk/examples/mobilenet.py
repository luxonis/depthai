import cv2
import depthai as dai
from depthai_sdk import OakCamera, AspectRatioResizeMode

with OakCamera() as oak:
    color = oak.create_camera('color', out='color')
    nn = oak.create_nn('mobilenet-ssd', color, out='dets')
    nn.config_nn(passthroughOut=True, aspectRatioResizeMode=AspectRatioResizeMode.STRETCH)

    oak.visualize([color, nn], fps=True)  # 1080P -> 720P
    oak.start(blocking=True)

import cv2
import depthai as dai
from depthai_sdk import OakCamera, AspectRatioResizeMode

with OakCamera(recording="people-tracking-above-02") as oak:
    color = oak.create_camera('color', out='color')
    nn = oak.create_nn('person-detection-0200', color, out='dets')
    nn.config_nn(passthroughOut=True, aspectRatioResizeMode=AspectRatioResizeMode.LETTERBOX)
    oak.visualize([color, nn], fps=True)  # 1080P -> 720P
    oak.start(blocking=True)

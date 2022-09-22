import cv2
import depthai as dai
from depthai_sdk import OakCamera, AspectRatioResizeMode

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    oak.visualize([nn.out, nn.out_passthrough])
    oak.show_graph()
    oak.start(blocking=True)

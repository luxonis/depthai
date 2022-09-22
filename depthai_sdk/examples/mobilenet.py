import cv2
import depthai as dai
from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color, spatial=True)
    oak.visualize([nn.out, nn.out_passthrough])
    oak.visualize(nn.out_spatials, scale=1/2)
    # oak.show_graph()
    oak.start(blocking=True)

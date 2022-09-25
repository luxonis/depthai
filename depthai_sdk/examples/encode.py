import cv2
import depthai as dai
from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color, tracker=True, spatial=True)
    nn.config_tracker(trackLabels=['person'])
    oak.visualize(nn.out_tracker)
    oak.visualize(nn.out, scale=2/3)
    # oak.show_graph()
    oak.start(blocking=True)

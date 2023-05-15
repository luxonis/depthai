import logging

import cv2

from depthai_sdk import OakCamera, set_logging_level
from depthai_sdk.classes import TwoStagePacket

set_logging_level(logging.DEBUG)


def callback(packet: TwoStagePacket):
    visualizer = packet.visualizer
    frame = visualizer.draw(packet.frame)
    cv2.imshow('Hand recognition', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    det = oak.create_nn('palm_detection_128x128', color)
    det.config_nn(resize_mode='crop')

    hand_landmark_nn = oak.create_nn('hand-landmark-lite', input=det)

    # Visualize detections on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(hand_landmark_nn)
    oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)

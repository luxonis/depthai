import cv2

from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket
from depthai_sdk.visualize.visualizer_helper import FramePosition, VisualizerHelper


def callback(packet: DetectionPacket):
    visualizer = packet.visualizer
    print('Detections:', packet.img_detections.detections)
    VisualizerHelper.print(packet.frame, 'BottomRight!', FramePosition.BottomRight)
    frame = visualizer.draw(packet.frame)
    cv2.imshow('Visualizer', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)

    oak.visualize([nn], fps=True, callback=callback)
    oak.start(blocking=True)

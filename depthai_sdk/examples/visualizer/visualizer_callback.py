import cv2
from depthai_sdk import OakCamera, VisualizerHelper, DetectionPacket, Visualizer
from depthai_sdk.visualize.visualizer_helper import FramePosition

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)

    def cb(packet: DetectionPacket, visualizer: Visualizer):
        print('Detections:', packet.img_detections.detections)
        VisualizerHelper.print(packet.frame, 'BottomRight!', FramePosition.BottomRight)
        cv2.imshow('frame', packet.frame)

    oak.visualize([nn], fps=True, callback=cb)
    oak.start(blocking=True)

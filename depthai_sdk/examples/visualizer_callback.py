import cv2
from depthai_sdk import OakCamera, BaseVisualizer, FramePosition, DetectionPacket

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)

    def cb(packet: DetectionPacket):
        print('Detections:', packet.imgDetections.detections)
        BaseVisualizer.print(packet.frame, 'BottomRight!', FramePosition.BottomRight)
        cv2.imshow('frame', packet.frame)

    oak.visualize([nn], fps=True, callback=cb)
    oak.start(blocking=True)

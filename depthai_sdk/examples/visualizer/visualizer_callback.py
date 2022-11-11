import cv2

from depthai_sdk import OakCamera, VisualizerHelper, DetectionPacket
from depthai_sdk.callback_context import CallbackContext
from depthai_sdk.visualize.visualizer_helper import FramePosition


def callback(ctx: CallbackContext):
    packet: DetectionPacket = ctx.packet
    print('Detections:', packet.img_detections.detections)
    VisualizerHelper.print(packet.frame, 'BottomRight!', FramePosition.BottomRight)
    cv2.imshow('frame', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)

    oak.visualize([nn], fps=True, callback=callback)
    oak.start(blocking=True)

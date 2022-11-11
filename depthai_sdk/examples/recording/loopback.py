import cv2

from depthai_sdk import OakCamera, VisualizerHelper
from depthai_sdk.callback_context import VisualizeContext
from depthai_sdk.visualize.visualizer_helper import FramePosition

i = 0
FPS = 30


def callback(ctx: VisualizeContext):
    global i

    packet = ctx.packet
    recorders = ctx.recorders

    VisualizerHelper.print(packet.frame, 'BottomRight!', FramePosition.BottomRight)
    cv2.imshow('frame', packet.frame)

    # recorder
    if i == FPS * 10:
        recorders[packet.name.split(';')[-1]].get_last(10)

    i += 1


with OakCamera() as oak:
    color = oak.create_camera('color', fps=FPS)
    nn = oak.create_nn('mobilenet-ssd', color)

    oak.visualize(nn, fps=True, callback=callback, record='recordings', keep_last_seconds=10)
    oak.start(blocking=True)

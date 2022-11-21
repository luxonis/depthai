import cv2

from depthai_sdk import OakCamera, VisualizerHelper
from depthai_sdk.callback_context import CallbackContext
from depthai_sdk.visualize.visualizer_helper import FramePosition

i = 0
FPS = 30


def callback(ctx: CallbackContext):
    global i

    packet = ctx.packet
    recorders = ctx.recorders

    VisualizerHelper.print(packet.frame, 'BottomRight!', FramePosition.BottomRight)
    cv2.imshow('frame', packet.frame)

    # Save the last 3 seconds after script every 10 seconds
    if i == FPS * 10:
        recorders['color'].get_last(3)
        i = 0
    i += 1


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', fps=FPS)
    oak.visualize(color.out.main, fps=True, callback=callback, record_path='recordings', keep_last_seconds=10)
    oak.start(blocking=True)

from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import FramePacket
import cv2

def cb(packet: FramePacket):
    cv2.imshow('Color frames from cb', packet.frame)

with OakCamera() as oak:
    color = oak.create_camera('color')
    oak.callback(
        color, # Outputs whose packets we want to receive via callback
        callback=cb, # Callback function
        main_thread=True # Whether to call the callback in the main thread. For OpenCV's imshow to work, it must be called in the main thread.
    )

    oak.start(blocking=True)
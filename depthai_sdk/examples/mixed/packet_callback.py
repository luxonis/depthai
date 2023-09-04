from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import FramePacket
from datetime import timedelta
from typing import Dict
import cv2

def cb_1(packet: FramePacket):
    # Called from main thread, so we can call cv2.imshow
    cv2.imshow('Color frames from cb', packet.frame)

def cb_2(packets: Dict[str, FramePacket]):
    print(packets)
    # Sycned packets.
    ts_color = packets['color'].get_timestamp()
    ts_left = packets['left'].get_timestamp()
    ts_imu = packets['imu'].get_timestamp()
    print(f"---- New synced packets. Diff between color and left: {abs(ts_color-ts_left) / timedelta(milliseconds=1)} ms, color and IMU: {abs(ts_imu-ts_color) / timedelta(milliseconds=1)} ms")

    for name, packet in packets.items():
        print(f'Packet {name}, timestamp: {packet.get_timestamp()}, Seq number: {packet.get_sequence_num()}')

with OakCamera() as oak:
    color = oak.create_camera('color', fps=32)
    left = oak.create_camera('left', fps=30)
    right = oak.create_camera('right', fps=30)
    imu = oak.create_imu()

    oak.callback(
        color, # Outputs whose packets we want to receive via callback
        callback=cb_1, # Callback function
        main_thread=True # Whether to call the callback in the main thread. For OpenCV's imshow to work, it must be called in the main thread.
    )

    cb_handler = oak.callback(
        [left, right, color, imu],
        callback=cb_2,
        main_thread=False # Will be called from a different thread, instead of putting packets into queue and waiting for main thread to pick it up.
    )
    # Timestamp syncing all 3 streams. We selected (1000/30) / 2 as threshold_ms, because
    # left/right are slower (30FPS), so threshold should be about 16ms. This means SDK will discard some
    # color packets (2 per second), but we will have synced frames.
    cb_handler.configure_syncing(threshold_ms=int((1000/30) / 2))

    # oak.show_graph()
    oak.start(blocking=True)
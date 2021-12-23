#!/usr/bin/env python3
import os
import platform
import depthai as dai
import time

if os.name == 'nt':
    print("This app is temporarily disabled on Windows system due to an issue with USB descriptors. We are working on resolving this issue")
    raise SystemExit(0)

if platform.machine() == 'aarch64':
    print("This app is temporarily disabled on AARCH64 systems due to an issue with stream preview. We are working on resolving this issue")
    raise SystemExit(0)

enable_4k = True  # Will downscale 4K -> 1080p

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setInterleaved(False)

if enable_4k:
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setIspScale(1, 2)
else:
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create an UVC (USB Video Class) output node. It needs 1920x1080, NV12 input
uvc = pipeline.createUVC()
cam_rgb.video.link(uvc.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    print("\nDevice started, please keep this process running")
    print("and open an UVC viewer. Example on Linux:")
    print("    guvcview -d /dev/video0")
    print("\nTo close: Ctrl+C")

    # Doing nothing here, just keeping the host feeding the watchdog
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break

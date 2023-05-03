#!/usr/bin/env python3
import platform
import depthai as dai
import time
import sys
import signal
import threading
from depthai_sdk.managers import arg_manager

args = arg_manager.parseArgs()

if platform.machine() == 'aarch64':
    print("This app is temporarily disabled on AARCH64 systems due to an issue with stream preview. We are working on resolving this issue", file=sys.stderr)
    raise SystemExit(1)

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

# Terminate app handler
quitEvent = threading.Event()
signal.signal(signal.SIGTERM, lambda *_args: quitEvent.set())

# Pipeline defined, now the device is connected to
with dai.Device(pipeline, usb2Mode=False) as device:
    if device.getDeviceInfo().protocol == dai.XLinkProtocol.X_LINK_USB_VSC and device.getUsbSpeed() not in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS):
        print("This app is temporarily disabled with USB2 connection speed due to known issue. We're working on resolving it. In the meantime, please try again with a USB3 cable/port for the device connection", file=sys.stderr)
        raise SystemExit(1)
    print("\nDevice started, please keep this process running")
    print("and open an UVC viewer. Example on Linux:")
    print("    guvcview -d /dev/video0")
    print("\nTo close: Ctrl+C")

    # Doing nothing here, just keeping the host feeding the watchdog
    while not quitEvent.is_set():
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break

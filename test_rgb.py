#!/usr/bin/env python3


# import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Properties
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
# camLeft.set
# camLeft.setInterleaved(False)
# camLeft.setColorOrder(dai.ColorCameraProperties.ColorOrder.MONO)

# Linking
camRgb.preview.link(xoutRgb.input)
camLeft.out.link(xoutLeft.input)
camRight.out.link(xoutRight.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name='right', maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        inLeft = qLeft.get()
        inRight = qRight.get()

        # Retrieve 'bgr' (opencv format) frame
        # cv2.imshow("rgb", inRgb.getCvFrame())
        # cv2.imshow("left", inLeft.getCvFrame())
        # cv2.imshow("right", inLeft.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
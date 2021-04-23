#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import math

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
imu = pipeline.createIMU()
xlinkOut = pipeline.createXLinkOut()
xlinkOut.setStreamName("imu")

# Link plugins CAM -> XLINK
imu.out.link(xlinkOut.input)

path = '/home/sachin/Downloads/depthai_imu_fw_update.cmd'
# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline, path) as device:
    # Start pipeline
    baseTs = time.monotonic()
    device.startPipeline()
    print("Starting pipeline...")

    # Output queue for imu bulk packets
    imuQueue = device.getOutputQueue(name="imu", maxSize=4, blocking=False)

    while True:
        imuPacket = imuQueue.get()  # blocking call, will wait until a new data has arrived

        imuDatas = imuPacket.imuDatas
        for imuData in imuDatas:
            dur = imuData.ts.getTimestamp()
            # TODO substract base time
            ffs = "{: .06f}"
            accelLength = math.sqrt(imuData.accelerometer.x**2 + imuData.accelerometer.y**2 + imuData.accelerometer.z**2)
            
            print(f"Timestamp: {dur}")
            print(f"Accel: {ffs.format(imuData.accelerometer.x)} {ffs.format(imuData.accelerometer.y)} {ffs.format(imuData.accelerometer.z)}, length {ffs.format(accelLength)}")
            print(f"Gyro:  {ffs.format(imuData.gyro.x)} {ffs.format(imuData.gyro.y)} {ffs.format(imuData.gyro.z)} ")

        if cv2.waitKey(1) == ord('q'):
            break

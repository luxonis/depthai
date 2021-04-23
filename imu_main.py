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
    count = 0
    while count < 100:
        imuPacket = imuQueue.get()  # blocking call, will wait until a new data has arrived
        imuDatas = imuPacket.imuDatas
        count += 1
        

#!/usr/bin/env python3

import cv2
from depthai_sdk import OakCamera, Visualizer, FramePosition, DetectionPacket

with OakCamera(recording='people-images-01') as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('person-detection-retail-0013', color)
    oak.replay.setFps(3)

    def cb(packet: DetectionPacket):
        num = len(packet.detections)
        print('New msgs! Number of people detected:', num)
        Visualizer.print(packet.frame, f"Number of people: {num}", FramePosition.BottomMid)
        cv2.imshow(f'frame {packet.name}', packet.frame)

    oak.visualize(nn, fps=True, callback=cb)
    # oak.show_graph()
    oak.start(blocking=True)

#!/usr/bin/env python3

import cv2

from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket
from depthai_sdk.visualize.configs import TextPosition


def callback(packet: DetectionPacket):
    visualizer = packet.visualizer
    num = len(packet.img_detections.detections)
    print('New msgs! Number of people detected:', num)

    visualizer.add_text(f"Number of people: {num}", position=TextPosition.TOP_MID)
    visualizer.draw(packet.frame)
    cv2.imshow(f'frame {packet.name}', packet.frame)


with OakCamera(replay='people-images-01') as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('person-detection-retail-0013', color)
    oak.replay.set_fps(0.5)

    oak.visualize(nn, callback=callback)
    # oak.show_graph()
    oak.start(blocking=True)

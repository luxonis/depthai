#!/usr/bin/env python3

import cv2
from depthai_sdk import OakCamera, BaseVisualizer, FramePosition

with OakCamera(recording='people-images-01') as oak:
    color = oak.create_camera('color', out='color')
    nn = oak.create_nn('person-detection-retail-0013', color, out='dets')
    nn.config_nn(passthroughOut=True)
    oak.replay.setFps(3)

    def cb(msgs, frame):
        num = len(msgs['dets'].detections)
        print('New msgs! Number of people detected:', num)
        BaseVisualizer.print(frame, f"Number of people: {num}", FramePosition.BottomMid)
        cv2.imshow(f'frame {frame.shape[0] }', frame)

    oak.visualize([color, nn], fps=True, callback=cb)
    oak.start(blocking=True)

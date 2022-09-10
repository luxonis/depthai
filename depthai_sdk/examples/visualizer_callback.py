import cv2
from depthai_sdk import OakCamera, BaseVisualizer, FramePosition

with OakCamera() as oak:
    color = oak.create_camera('color', out='color')
    nn = oak.create_nn('mobilenet-ssd', color, out='dets')

    def cb(msgs, frame):
        print('New msgs! Detections: ', [det.label for det in msgs['dets'].detections])
        BaseVisualizer.print(frame, 'TopLeft!', FramePosition.TopLeft)
        BaseVisualizer.print(frame, 'Middle!', FramePosition.Mid)
        BaseVisualizer.print(frame, 'BottomRight!', FramePosition.BottomRight)
        cv2.imshow('frame', frame)

    oak.visualize([color, nn], fps=True, callback=cb)
    oak.start(blocking=True)

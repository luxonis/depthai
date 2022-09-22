import cv2
from depthai_sdk import OakCamera

with OakCamera() as oak:
    left = oak.create_camera('left', resolution='400p', fps=117)
    right = oak.create_camera('right', resolution='400p', fps=117)
    oak.visualize([left, right], fps=True)

    oak.start(blocking=True)

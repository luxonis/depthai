from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    left = oak.create_camera('left')
    right = oak.create_camera('right')
    oak.visualize([color, left, right], fps=True)
    oak.start(blocking=True)

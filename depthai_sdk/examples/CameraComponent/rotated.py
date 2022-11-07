from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', rotation=90)
    left = oak.create_camera('left', resolution='400p', rotation=180)
    right = oak.create_camera('right', resolution='400p', rotation=270)
    oak.visualize([color, left, right], fps=True)
    oak.start(blocking=True)

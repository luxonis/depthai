from depthai_sdk import OakCamera

with OakCamera(rotation=90) as oak:
    color = oak.create_camera('color', resolution='1080p')
    left = oak.create_camera('left', resolution='400p')
    right = oak.create_camera('right', resolution='400p')
    oak.visualize([color, left, right], fps=True)
    oak.start(blocking=True)

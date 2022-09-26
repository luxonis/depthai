from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    oak.visualize(color)
    oak.start(blocking=True)

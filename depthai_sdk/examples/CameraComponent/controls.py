from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.enable_controls()
    oak.visualize(color, fps=True)
    oak.start(blocking=True)

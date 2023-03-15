from depthai_sdk import OakCamera

with OakCamera() as oak:
    cams = oak.create_all_cameras()

    oak.visualize(cams)
    oak.start(blocking=True)

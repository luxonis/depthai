from depthai_sdk import OakCamera

with OakCamera(rotation=90) as oak:
    all_cams = oak.create_all_cameras()
    oak.visualize(all_cams, fps=True)
    oak.start(blocking=True)

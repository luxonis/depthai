from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', encode='h265')

    oak.visualize(color.out.encoded, fps=True, scale=2/3)
    # By default, it will stream non-encoded frames
    oak.visualize(color, fps=True, scale=2/3)
    oak.start(blocking=True)

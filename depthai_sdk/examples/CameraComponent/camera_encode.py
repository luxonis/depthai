from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    encoder = oak.create_encoder(color, codec='h265')

    oak.visualize(encoder, fps=True, scale=2/3)
    # By default, it will stream non-encoded frames
    oak.visualize(color, fps=True, scale=2/3)
    oak.start(blocking=True)

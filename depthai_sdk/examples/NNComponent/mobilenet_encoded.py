from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', encode='mjpeg', fps=10)
    nn = oak.create_nn('mobilenet-ssd', color, spatial=True)
    oak.visualize([nn.out.encoded])
    oak.start(blocking=True)

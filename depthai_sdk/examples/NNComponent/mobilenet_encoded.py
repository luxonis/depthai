from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', encode='mjpeg', fps=10)

    nn = oak.create_nn('mobilenet-ssd', color, spatial=True)  # spatial flag indicates that we want to get spatial data

    oak.visualize([nn.out.encoded])  # Display encoded output
    oak.start(blocking=True)

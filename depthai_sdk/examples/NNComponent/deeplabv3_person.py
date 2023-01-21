from depthai_sdk import OakCamera


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    nn = oak.create_nn('deeplabv3_person', color)
    nn.config_nn(resize_mode='letterbox')

    visualizer = oak.visualize([nn, nn.out.passthrough], fps=True)
    oak.start(blocking=True)

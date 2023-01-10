from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color, spatial=True, tracker=True)
    nn.config_nn(aspect_ratio_resize_mode='stretch')
    visualizer = oak.visualize([nn.out.tracker], fps=True)
    visualizer.tracking(speed=True)

    oak.start(blocking=True)

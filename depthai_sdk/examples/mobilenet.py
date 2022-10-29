from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color, spatial=True)
    oak.visualize([nn.out.main, nn.out.passthrough], fps=True, scale=2/3)
    # oak.show_graph()
    oak.start(blocking=True)

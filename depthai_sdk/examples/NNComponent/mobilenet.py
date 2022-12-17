from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color, spatial=True)
    oak.visualize([nn.out.main, nn.out.spatials], fps=True)
    oak.start(blocking=True)

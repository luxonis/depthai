from depthai_sdk import OakCamera

with OakCamera() as oak:
    left = oak.create_camera('left')
    nn = oak.create_nn('face-detection-retail-0004', left)
    oak.visualize([nn.out.main, nn.out.passthrough], scale=2/3, fps=True)
    oak.start(blocking=True)

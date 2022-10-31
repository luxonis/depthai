from depthai_sdk import OakCamera

with OakCamera(replay='https://www.youtube.com/watch?v=Y1jTEyb3wiI') as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('vehicle-detection-0202', color)
    oak.visualize([nn.out.passthrough], fps=True)
    oak.visualize(nn, scale=2 / 3, fps=True)
    oak.start(blocking=True)

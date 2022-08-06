from depthai_sdk import Camera

with Camera() as cam:
    color = cam.create_camera('color', out='color')
    nn = cam.create_nn('yolo-v3-tf', color, out='dets')
    cam.create_visualizer([color, nn], scale=2/3, fps=True) # 1080P -> 720P
    cam.start(blocking=True)
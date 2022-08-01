from depthai_sdk import Camera

with Camera(recording='depth-people-counting-01') as cam:
    color = cam.create_camera('color', out='color')
    nn = cam.create_nn('vehicle-detection-adas-0002', color, out='dets')
    cam.create_visualizer([color, nn], scale=2/3) # 1080P -> 720P
    cam.start(blocking=True)
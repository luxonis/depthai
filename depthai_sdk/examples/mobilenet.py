from depthai_sdk import Camera

# with Camera(recording='cars-california-01') as cam:
with Camera() as cam:
    color = cam.create_camera('color', out='color')
    nn = cam.create_nn('vehicle-detection-adas-0002', color, out='dets')
    nn.configNn(True)
    cam.create_visualizer([color, nn]) # 1080P -> 720P
    cam.start(blocking=True)
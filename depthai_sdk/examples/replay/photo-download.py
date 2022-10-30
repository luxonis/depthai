from depthai_sdk import OakCamera

with OakCamera(replay='https://images.pexels.com/photos/3184398/pexels-photo-3184398.jpeg?w=800&h=600') as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('face-detection-retail-0004', color)
    oak.visualize([nn.out.passthrough, nn])
    oak.start(blocking=True)

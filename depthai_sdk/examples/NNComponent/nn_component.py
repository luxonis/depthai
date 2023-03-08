from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    # Supported models: https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/#sdk-supported-models
    nn = oak.create_nn('yolov6nr3_coco_640x352', color)
    oak.visualize([nn.out.main], fps=True)
    oak.visualize(nn.out.passthrough)
    oak.start(blocking=True)

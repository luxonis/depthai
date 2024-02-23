from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    # List of models that are supported out-of-the-box by the SDK:
    # https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/#sdk-supported-models
    nn = oak.create_nn('yolov5n_coco_416x416', color)
    nn.config_nn(resize_mode='stretch')
    oak.visualize([nn.out.main], fps=True)
    oak.visualize(nn.out.passthrough)
    oak.start(blocking=True)

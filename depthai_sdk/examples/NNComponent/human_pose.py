from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    # List of models that are supported out-of-the-box by the SDK:
    # https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/#sdk-supported-models
    human_pose_nn = oak.create_nn('human-pose-estimation-0001', color)

    oak.visualize(human_pose_nn)
    oak.start(blocking=True)

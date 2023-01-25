from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    human_pose_nn = oak.create_nn('human-pose-estimation-0001', color)
    oak.visualize(human_pose_nn)
    oak.start(blocking=True)

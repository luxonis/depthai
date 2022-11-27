from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', fps=30)

    nn = oak.create_nn('mobilenet-ssd')
    # Triggers, maybe wihtout .when(), so directly nn.detection().record()
    # nn.detection.when(lambda daiDetection: daiDetection.confidence < 0.8)
    # nn.detections.when(lambda daiDetections: len(daiDetections.detections) == 3)
    ## Result would be the decoded NN result (eg. SemanticSegmentation, ImgLandmarks, InstanceSegmentation...)
    ## Check if first landmark (eg. left eye) is in the second quadrant
    # landmark_nn.result.when(lambda result: result.landmarks[0][0] < 0.5 and  result.landmarks[0][1] < 0.5)

    ## Actions:
    nn.detections.record(
        condition_func=lambda dets: len(dets) == 2,
        dir_path='recordings/detections',
        stream=nn.out.main,
        offset=0,
        duration=10
    )

    oak.start(blocking=True)

    # def record( # Just use cv2.VideoWriter?
    #     path: str,
    #     stream: XoutFrames,
    #     frame_start_rec: int = 0, # From which frame before/after trigger should we start recording frames
    #     frame_end_rec: int = 10, # After how many frames after the last trigger should we stop recording
    #     callback: Callable = None # If callbakc returns False, frame won't get saved in the video
    # )
    #
    # def photo( # Use cv2.imwrite()
    #     path: str,
    #     stream: XoutFrames, # in the future, if CameraComponent and color camera, create still event?
    #     frame_photo: int = 0, # When - before/after the trigger - to take the photo (can be negative number as well)
    #     cooldown: int = 30, # Cooldown for taking another photo (in number of frames)
    #     callback: Callable = None # If callback returns False, photo won't get saved
    # )
    # We will add additional actions afterwards; like writing to txt file, saving to mcap, uploading to robothub for model retraining etc.
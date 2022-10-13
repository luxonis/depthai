from depthai_sdk import OakCamera, AspectRatioResizeMode, RecordType

with OakCamera(recording='cars-tracking-above-01') as oak:
    # Create color camera, add video encoder
    color = oak.create_camera('color', encode='H264')
    # Run Spatial MobileNet + track detections
    nn = oak.create_nn('vehicle-detection-0202', color, tracker=True)
    nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.STRETCH)
    # Visualize tracklets, show FPS, downscale frame
    oak.visualize([nn.out.tracker], scale=2/3, fps=True)
    # Visualize the NN passthrough frame + detections
    oak.visualize([nn.out.passthrough])
    # Record color H264 stream
    oak.record(color.out.encoded, './color-recording', RecordType.VIDEO)

    # Start the app in blocking mode
    oak.start(blocking=True)
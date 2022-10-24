from depthai_sdk import OakCamera, AspectRatioResizeMode, RecordType

# Download public depthai-recording
with OakCamera(recording='cars-tracking-above-01') as oak:
    # Create color camera, add video encoder
    color = oak.create_camera('color')
    # Download & run pretrained vehicle detection model and track detections
    nn = oak.create_nn('vehicle-detection-0202', color, tracker=True)
    nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.STRETCH)
    # Visualize tracklets, show FPS, downscale frame
    visualizer = oak.visualize([nn.out.tracker], record='./visualzerVideo.mp4')
    visualizer.tracking(line_thickness=3).text(auto_scale=True)
    # Start the app in blocking mode
    oak.start(blocking=True)
from depthai_sdk import OakCamera, ResizeMode

# Download public depthai-recording
with OakCamera(replay='cars-tracking-above-01') as oak:
    # Create color camera, add video encoder
    color = oak.create_camera('color')

    # Download & run pretrained vehicle detection model and track detections
    nn = oak.create_nn('vehicle-detection-0202', color, tracker=True)

    # Visualize tracklets, show FPS
    visualizer = oak.visualize(nn.out.tracker, fps=True, record_path='./car_tracking.avi')
    visualizer.tracking(line_thickness=5).text(auto_scale=True)
    # Start the app in blocking mode
    # oak.show_graph()
    oak.start(blocking=True)

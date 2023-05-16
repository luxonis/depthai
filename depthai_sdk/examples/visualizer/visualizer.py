from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import BboxStyle, TextPosition

with OakCamera() as oak:
    camera = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', camera)
    # Record visualized video into a mp4 file
    visualizer = oak.visualize(det.out.main, record_path='./test.mp4')
    # Chained methods for setting visualizer parameters
    visualizer.detections(  # Detection-related parameters
        color=(0, 255, 0),
        thickness=2,
        bbox_style=BboxStyle.RECTANGLE,  # Options: RECTANGLE, CORNERS, ROUNDED_RECTANGLE, ROUNDED_CORNERS
        label_position=TextPosition.MID,
    ).text(  # Text-related parameters
        font_color=(255, 255, 0),
        auto_scale=True
    ).output(  # General output parameters
        show_fps=True,
    ).tracking(  # Tracking-related parameters
        line_thickness=5
    )

    oak.start(blocking=True)

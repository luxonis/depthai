from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import BboxStyle, TextPosition

with OakCamera() as oak:
    camera = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', camera, tracker=True, spatial=True)

    visualizer = oak.visualize(det.out.tracker)
    visualizer.configure_bbox(
        color=(0, 255, 0),
        thickness=2,
        bbox_style=BboxStyle.RECTANGLE,
        label_position=TextPosition.MID,
    ).configure_text(
        font_color=(255, 255, 0)
    ).configure_output(
        show_fps=True,
    ).configure_tracking(
        line_thickness=5
    )

    oak.start(blocking=True)

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import BboxStyle, TextPosition

with OakCamera() as oak:
    camera = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', camera)

    visualizer = oak.visualize(det.out.main)
    visualizer.detections(
        color=(0, 255, 0),
        thickness=2,
        bbox_style=BboxStyle.RECTANGLE,
        label_position=TextPosition.MID,
    ).text(
        font_color=(255, 255, 0),
        auto_scale=True
    ).output(
        show_fps=True,
    ).tracking(
        line_thickness=5
    )

    oak.start(blocking=True)

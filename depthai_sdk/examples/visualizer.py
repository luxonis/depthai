from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import BboxStyle, TextPosition

with OakCamera() as oak:
    camera = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', camera)

    visualizer = oak.visualize(det.out.main)
    visualizer.configure_bbox(
        color=(0, 255, 0),
        thickness=2,
        bbox_style=BboxStyle.RECTANGLE,
        label_position=TextPosition.MID,
    ).configure_text(
        font_color=(255, 255, 0)
    )

    visualizer2 = oak.visualize(det.out.main)
    visualizer2.configure_bbox(
        fill_transparency=0.7,
        thickness=3,
        bbox_style=BboxStyle.CORNERS,
        label_position=TextPosition.BOTTOM_RIGHT,
    ).configure_text(
        font_color=(0, 0, 0),
        font_scale=2.0,
        font_thickness=3
    )

    oak.start(blocking=True)

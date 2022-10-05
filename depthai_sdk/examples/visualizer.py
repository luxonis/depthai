from depthai_sdk.visualize.configs import BboxStyle

from depthai_sdk import OakCamera

with OakCamera() as oak:
    camera = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', camera)

    visualizer = oak.visualize(det.out.main)
    visualizer.configure_bbox(
        color=(0, 255, 0),
        thickness=2,
        bbox_style=BboxStyle.RECTANGLE,
    ).configure_text(
        font_color=(0, 255, 0)
    )

    visualizer2 = oak.visualize(det.out.main)
    visualizer2.configure_bbox(
        fill_transparency=0.7,
        thickness=3,
        bbox_style=BboxStyle.CORNERS,
        hide_label=True
    )

    visualizer3 = oak.visualize(det.out.main)
    visualizer3.configure_bbox(
        fill_transparency=0.5,
        box_roundness=25,
        thickness=1,
        bbox_style=BboxStyle.ROUNDED_RECTANGLE,
        hide_label=True
    )

    visualizer4 = oak.visualize(det.out.main)
    visualizer4.configure_bbox(
        fill_transparency=0,
        box_roundness=15,
        thickness=1,
        bbox_style=BboxStyle.ROUNDED_CORNERS,
    ).configure_text(
        font_color=(0, 255, 255)
    )

    oak.start(blocking=True)

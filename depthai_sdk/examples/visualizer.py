from depthai_sdk import NewVisualizer
from depthai_sdk import OakCamera, DetectionPacket


with OakCamera() as oak:
    camera = oak.create_camera('color')

    det = oak.create_nn('face-detection-retail-0004', camera)

    visualizer = oak.visualize(det.out.main)
    visualizer.configure_bbox(
        color=(0, 255, 0),
        box_roundness=0.5,
        thickness=2
    )

    visualizer2 = oak.visualize(det.out.main)
    visualizer2.configure_bbox(
        fill_transparency=0.5,
        thickness=-1
    )

    oak.start(blocking=True)

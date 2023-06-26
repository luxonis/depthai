from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import FramePacket
import cv2

with OakCamera() as oak:
    color = oak.create_camera('color')
    q = oak.create_queue(color, max_size=5)

    oak.show_graph()
    oak.start()

    while oak.running():
        oak.poll()
        p: FramePacket = q.get(block=True)
        cv2.imshow(p.name, p.frame)
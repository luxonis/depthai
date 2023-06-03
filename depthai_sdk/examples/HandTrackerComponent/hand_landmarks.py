from depthai_sdk import OakCamera
from depthai_sdk.components.hand_tracker.renderer import HandTrackerRenderer
from depthai_sdk.classes.packets import HandTrackerPacket
import cv2

with OakCamera() as oak:
    color = oak.create_camera('color')

    handtracker = oak.create_hand_tracker(color)

    render = HandTrackerRenderer(handtracker)
    def cb(packet: HandTrackerPacket):
        render.draw(packet.color_frame, packet.hands)
        cv2.imshow("Hand tracking", render.frame)

    oak.callback(handtracker, cb)
    oak.visualize(handtracker.out.palm_detection)
    oak.visualize(handtracker.out.palm_crop)

    # oak.show_graph()
    oak.start(blocking=True)

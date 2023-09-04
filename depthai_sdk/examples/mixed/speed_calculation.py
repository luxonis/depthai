import cv2

from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import TrackerPacket


def callback(packet: TrackerPacket):
    for obj_id, tracklets in packet.tracklets.items():
        if len(tracklets) != 0:
            tracklet = tracklets[-1]
        if tracklet.speed is not None:
            print(f'Speed for object {obj_id}: {tracklet.speed:.02f} m/s, {tracklet.speed_kmph:.02f} km/h, {tracklet.speed_mph:.02f} mph')

    frame = packet.visualizer.draw(packet.decode())
    cv2.imshow('Speed estimation', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    stereo = oak.create_stereo('800p')
    stereo.config_stereo(subpixel=False, lr_check=True)

    nn = oak.create_nn('face-detection-retail-0004', color, spatial=stereo, tracker=True)
    nn.config_tracker(calculate_speed=True)

    visualizer = oak.visualize(nn.out.tracker, callback=callback, fps=True)
    visualizer.tracking(show_speed=True).text(auto_scale=True)

    oak.start(blocking=True)

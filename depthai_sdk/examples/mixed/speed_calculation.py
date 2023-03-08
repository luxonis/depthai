import cv2

from depthai_sdk import OakCamera


def callback(packet):
    for detection in packet.detections:
        print(f'Speed: {detection.speed:.02f} m/s, {detection.speed_kmph:.02f} km/h, {detection.speed_mph:.02f} mph')

    frame = packet.visualizer.draw(packet.frame)
    cv2.imshow('Speed estimation', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    stereo = oak.create_stereo('800p')
    stereo.config_stereo(subpixel=False, lr_check=True)

    nn = oak.create_nn('face-detection-retail-0004', color, spatial=stereo, tracker=True)
    nn.config_nn(resize_mode='stretch')

    visualizer = oak.visualize(nn.out.tracker, callback=callback, fps=True)
    visualizer.tracking(speed=True).text(auto_scale=True)

    oak.start(blocking=True)

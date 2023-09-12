import cv2
import numpy as np

from depthai_sdk import OakCamera
from depthai_sdk.classes import TwoStagePacket
from depthai_sdk.visualize.configs import TextPosition


def callback(packet: TwoStagePacket):
    visualizer = packet.visualizer
    for det, rec in zip(packet.detections, packet.nnData):
        age = int(float(np.squeeze(np.array(rec.getLayerFp16('age_conv3')))) * 100)
        gender = np.squeeze(np.array(rec.getLayerFp16('prob')))
        gender_str = "Woman" if gender[0] > gender[1] else "Man"

        visualizer.add_text(f'{gender_str}\nAge: {age}',
                            bbox=packet.bbox.get_relative_bbox(det.bbox),
                            position=TextPosition.BOTTOM_RIGHT)

    frame = visualizer.draw(packet.frame)
    cv2.imshow('Age-gender estimation', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', color)
    det.config_nn(resize_mode='crop')

    age_gender = oak.create_nn('age-gender-recognition-retail-0013', input=det)
    # age_gender.config_multistage_nn(show_cropped_frames=True) # For debugging

    # Visualize detections on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(age_gender, callback=callback)
    oak.visualize(det.out.passthrough)
    oak.visualize(age_gender.out.twostage_crops)

    # oak.show_graph() # Show pipeline graph, no need for now
    oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)

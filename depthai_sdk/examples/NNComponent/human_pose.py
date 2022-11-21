import cv2

from depthai_sdk import OakCamera
from depthai_sdk.callback_context import CallbackContext

NN_HEIGHT, NN_WIDTH = 256, 456


def scale(p1, p2, scale_factor, offset_w):
    return int(p1 * scale_factor) + offset_w, int(p2 * scale_factor)


def callback(ctx: CallbackContext):
    packet = ctx.packet
    frame = packet.frame

    scale_factor = frame.shape[0] / NN_HEIGHT
    offset_w = int(frame.shape[1] - NN_WIDTH * scale_factor) // 2

    for person_landmarks in packet.img_detections.landmarks:
        for i, landmark in enumerate(person_landmarks):
            l1, l2 = landmark
            x1, y1 = scale(*l1, scale_factor, offset_w)
            x2, y2 = scale(*l2, scale_factor, offset_w)
            cv2.line(frame, (x1, y1), (x2, y2), packet.img_detections.colors[i], 3, cv2.LINE_AA)

    cv2.imshow('Human pose estimation', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    human_pose_nn = oak.create_nn('human-pose-estimation-0001', color)

    oak.callback(human_pose_nn, callback)
    oak.start(blocking=True)

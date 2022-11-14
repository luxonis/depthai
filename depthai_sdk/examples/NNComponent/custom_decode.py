import blobconverter
import cv2
import numpy as np
from depthai import NNData, ImgDetections

from depthai_sdk import OakCamera, Detections
from depthai_sdk.callback_context import VisualizeContext


def decode(nn_data: NNData):
    layer = nn_data.getFirstLayerFp16()
    results = np.array(layer).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)

    for result in results[0][0]:
        if result[2] > 0.5:
            dets.add(result[1], result[2], result[3:])

    return dets


def callback(ctx: VisualizeContext):
    packet = ctx.packet
    frame = packet.frame
    detections: Detections = packet.img_detections

    for bbox in detections.detections:
        scaled_bbox = np.array(bbox) * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
        scaled_bbox = np.int0(scaled_bbox)
        cv2.rectangle(frame, tuple(scaled_bbox[:2]), tuple(scaled_bbox[2:4]), (0, 0, 255), 2)

    cv2.imshow('Frame', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    nn_path = blobconverter.from_zoo(name='person-detection-0200', version='2022.1')
    nn = oak.create_nn(nn_path, color, decode_fn=decode)

    oak.visualize(nn, callback=callback)
    oak.start(blocking=True)

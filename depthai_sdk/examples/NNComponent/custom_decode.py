import blobconverter
import cv2
import numpy as np
from depthai import NNData

from depthai_sdk import OakCamera, Detections
from depthai_sdk.callback_context import CallbackContext


def decode(nn_data: NNData) -> Detections:
    layer = nn_data.getFirstLayerFp16()
    results = np.array(layer).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)

    for result in results[0][0]:
        if result[2] > 0.5:
            label = int(result[1])
            conf = result[2]
            bbox = result[3:]
            dets.add(label, conf, bbox)

    return dets


def callback(ctx: CallbackContext):
    packet = ctx.packet
    frame = packet.frame
    visualizer = ctx.visualizer

    frame = visualizer.draw(frame)
    cv2.imshow('Custom decode function', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    nn_path = blobconverter.from_zoo(name='person-detection-0200', version='2021.4')
    nn = oak.create_nn(nn_path, color, decode_fn=decode)

    oak.visualize(nn, callback=callback)
    oak.start(blocking=True)

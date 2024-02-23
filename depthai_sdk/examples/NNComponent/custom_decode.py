import blobconverter
import numpy as np
import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections


def decode(nn_data: dai.NNData) -> Detections:
    """
    Custom decode function for the NN component. Decode function has to accept NNData argument.
    The return type should preferably be a class that inherits from depthai_sdk.classes.GenericNNOutput,
    which support visualization. But this is not required, i.e. the function can return arbitrary type.

    The decoded output can be accessed from the packet object in the callback function via packet.img_detections.
    """
    layer = nn_data.getFirstLayerFp16()
    results = np.array(layer).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)
    for result in results[0][0]:
        if result[2] > 0.3:
            label = int(result[1])
            conf = result[2]
            bbox = result[3:]
            det = dai.ImgDetection()
            det.confidence = conf
            det.label = label
            det.xmin = bbox[0]
            det.ymin = bbox[1]
            det.xmax = bbox[2]
            det.ymax = bbox[3]
            dets.detections.append(det)

    return dets

with OakCamera() as oak:
    color = oak.create_camera('color')

    nn_path = blobconverter.from_zoo(name='person-detection-0200', version='2021.4', shaves=6)
    nn = oak.create_nn(nn_path, color, decode_fn=decode)

    oak.visualize(nn)
    oak.start(blocking=True)

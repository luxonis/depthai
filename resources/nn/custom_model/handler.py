import cv2
import numpy as np

from depthai_sdk import frameNorm


def decode(nnManager, packet):
    bboxes = np.array(packet.getFirstLayerFp16())
    bboxes = bboxes.reshape((bboxes.size // 7, 7))
    bboxes = bboxes[bboxes[:, 2] > 0.5]
    labels = bboxes[:, 1].astype(int)
    confidences = bboxes[:, 2]
    bboxes = bboxes[:, 3:7]
    return {
        "labels": labels,
        "confidences": confidences,
        "bboxes": bboxes
    }


decoded = ["unknown", "face"]


def draw(nnManager, data, frames):
    for name, frame in frames:
        if name == nnManager.source:
            for label, conf, raw_bbox in zip(*data.values()):
                bbox = frameNorm(frame, raw_bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, decoded[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

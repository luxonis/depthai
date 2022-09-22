from types import SimpleNamespace
from typing import Tuple, List, Union
import depthai as dai
import numpy as np
import math

class Detection():
    def __init__(self):
        pass

    # Original ImgDetection
    imgDetection: dai.ImgDetection
    label_str: str
    color: Tuple[int,int,int]
    # Normalized bounding box
    topLeft: Tuple[int,int]
    bottomRight: Tuple[int, int]


class TwoStageDetection(Detection):
    def __init__(self):
        super().__init__()

    nn_data: dai.NNData


class FramePacket:
    def __init__(self):
        pass
    name: str  # ImgFrame stream name
    imgFrame: dai.ImgFrame # Original depthai message
    frame: np.ndarray  # cv2 frame for visualization


class DetectionPacket(FramePacket):
    # Original depthai messages
    imgDetections: Union[dai.ImgDetections, dai.SpatialImgDetections]
    detections: List[Detection]

    def __init__(self,
                 name: str,
                 imgFrame: dai.ImgFrame,
                 imgDetections: Union[dai.ImgDetections, dai.SpatialImgDetections]):
        super().__init__()
        self.name = name
        self.imgFrame = imgFrame
        self.imgDetections = imgDetections
        self.frame = self.imgFrame.getCvFrame()
        self.detections = []

    def isSpatialDetection(self) -> bool:
        return isinstance(self.imgDetections, dai.SpatialImgDetections)

    @staticmethod
    def spatialsText(detection: dai.SpatialImgDetection):
        spatials = detection.spatialCoordinates
        return SimpleNamespace(
            x = "X: " + ("{:.1f}m".format(spatials.x / 1000) if not math.isnan(spatials.x) else "--"),
            y = "Y: " + ("{:.1f}m".format(spatials.y / 1000) if not math.isnan(spatials.y) else "--"),
            z = "Z: " + ("{:.1f}m".format(spatials.z / 1000) if not math.isnan(spatials.z) else "--"),
        )

    def add_detection(self, img_det: dai.ImgDetection, bbox: np.ndarray, txt:str, color, _=None):
        det = Detection()
        det.imgDetection = img_det
        det.label_str = txt
        det.color = color
        det.topLeft = (bbox[0], bbox[1])
        det.bottomRight = (bbox[2], bbox[3])
        self.detections.append(det)


class TwoStagePacket(FramePacket):
    # Original depthai messages
    imgDetections: dai.ImgDetections
    detections: List[TwoStageDetection]
    nnData: List[dai.NNData]
    labels: List[int] = None
    _cntr: int = 0 # Label counter

    def __init__(self, name: str,
                 imgFrame: dai.ImgFrame,
                 imgDetections: dai.ImgDetections,
                 nnData: List[dai.NNData],
                 labels: List[int]):
        super().__init__()
        self.detections = []
        self.name = name
        self.imgFrame = imgFrame
        self.imgDetections = imgDetections
        self.frame = self.imgFrame.getCvFrame()
        self.nnData = nnData
        self.labels = labels

    def isSpatialDetection(self) -> bool:
        return isinstance(self.imgDetections, dai.SpatialImgDetections)

    def add_detection(self, img_det: dai.ImgDetection, bbox: np.ndarray, txt:str, color):
        det = TwoStageDetection()
        det.imgDetection = img_det
        det.color = color
        det.topLeft = (bbox[0], bbox[1])
        det.bottomRight = (bbox[2], bbox[3])

        # Append the second stage NN result to the detection
        if self.labels is None or img_det.label in self.labels:
            det.nn_data = self.nnData[self._cntr]
            self._cntr += 1

        self.detections.append(det)


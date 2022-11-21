"""
General (standarized) NN outputs, to be used for higher-level abstractions (eg. automatic visualization of results).
"SDK supported NN models" will have to have standard NN output, so either dai.ImgDetections, or one of the outputs
below. If the latter, model json config will incldue handler.py logic for decoding to the standard NN output.
These will be integrated into depthai-core, bonus points for on-device decoding of some popular models.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Any

import numpy as np
from depthai import NNData, ImgDetections, ImgDetection


class GenericNNOutput:
    """
    Generic NN output, to be used for higher-level abstractions (eg. automatic visualization of results).
    """

    def __init__(self, nn_data: NNData):
        self.nn_data = nn_data


# First we have Object detection results, which are already standarized with dai.ImgDetections

@dataclass
class Detections(ImgDetections, GenericNNOutput):
    """
    Detection results containing bounding boxes, labels and confidences.
    """

    def __init__(self, nn_data: NNData):
        ImgDetections.__init__(self)
        GenericNNOutput.__init__(self, nn_data)

    def add(self, label: int, confidence: float, bbox: Tuple[float, ...]) -> None:
        det = ImgDetection()
        det.label = label
        det.confidence = confidence
        det.xmin = bbox[0]
        det.ymin = bbox[1]
        det.xmax = bbox[2]
        det.ymax = bbox[3]
        self.detections = [*self.detections, det]


@dataclass
class SemanticSegmentation(GenericNNOutput):  # In core, extend from NNData
    """
    Semantic segmentation results, with a mask for each class.

    Examples: `DeeplabV3`, `Lanenet`, `road-semgentation-adas-0001`.
    """

    def __init__(self, nn_data: NNData, mask: List[np.ndarray]):
        super().__init__(nn_data)
        self.mask = mask

    mask: List[np.ndarray] = field(default_factory=list)  # 2D np.array for each class


@dataclass
class ImgLandmarks(GenericNNOutput):  # In core, extend from NNData
    """
    Landmarks results, with a list of landmarks and pairs of landmarks to draw lines between.

    Examples: `human-pose-estimation-0001`, `openpose2`, `facial-landmarks-68`, `landmarks-regression-retail-0009`.
    """

    def __init__(self,
                 nn_data: NNData,
                 landmarks: List[List[Any]] = None,
                 pairs: List[Tuple[int, int]] = None,
                 colors: List[Tuple[int, int, int]] = None):
        super().__init__(nn_data)
        self.landmarks = landmarks
        self.pairs = pairs
        self.colors = colors

    landmarks: List[List[Any]] = field(default_factory=list)
    pairs: List[Tuple[int, int]] = None  # Pairs of landmarks, to draw lines between them
    colors: List[Tuple[int, int, int]] = None  # Color for each landmark (eg. both elbows are in the same color)


@dataclass
class InstanceSegmentation(GenericNNOutput):
    """
    Instance segmentation results, with a mask for each instance.
    """

    def __init__(self, nn_data: NNData, masks: List[np.ndarray], labels: List[int]):
        super().__init__(nn_data)

    masks: List[np.ndarray] = field(default_factory=list)  # 2D np.array for each instance
    labels: List[int] = field(default_factory=list)  # Class label for each instance

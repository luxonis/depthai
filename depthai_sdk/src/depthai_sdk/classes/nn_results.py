"""
General (standarized) NN outputs, to be used for higher-level abstractions (eg. automatic visualization of results).
"SDK supported NN models" will have to have standard NN output, so either dai.ImgDetections, or one of the outputs
below. If the latter, model json config will incldue handler.py logic for decoding to the standard NN output.
These will be integrated into depthai-core, bonus points for on-device decoding of some popular models.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Any

import numpy as np
from depthai import NNData


class GenericNNOutput:
    """
    Generic NN output, to be used for higher-level abstractions (eg. automatic visualization of results).
    """
    def __init__(self, nn_data):
        self.nn_data = nn_data


# First we have Object detection results, which are already standarized with dai.ImgDetections

@dataclass
class Detections(GenericNNOutput):
    """
    Detection results containing bounding boxes, labels and confidences.
    """

    def __init__(self, nn_data: NNData):
        super().__init__(nn_data)
        self.detections = []
        self.labels = []
        self.confidences = []

    detections: List[Tuple[float, ...]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def add(self, label: str, confidence: float, bbox: Tuple[float, ...]) -> None:
        self.labels.append(label)
        self.confidences.append(confidence)
        self.detections.append(bbox)


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

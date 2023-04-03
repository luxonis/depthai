"""
General (standarized) NN outputs, to be used for higher-level abstractions (eg. automatic visualization of results).
"SDK supported NN models" will have to have standard NN output, so either dai.ImgDetections, or one of the outputs
below. If the latter, model json config will incldue handler.py logic for decoding to the standard NN output.
These will be integrated into depthai-core, bonus points for on-device decoding of some popular models.
"""
from dataclasses import dataclass
from typing import List, Tuple, Any

import numpy as np
from depthai import NNData, ImgDetection


class GenericNNOutput:
    """
    Generic NN output, to be used for higher-level abstractions (eg. automatic visualization of results).
    """

    def __init__(self, nn_data: NNData):
        self.nn_data = nn_data


# First we have Object detection results, which are already standarized with dai.ImgDetections

@dataclass
class Detections(GenericNNOutput):
    """
    Detection results containing bounding boxes, labels and confidences. Optionally can contain rotation angles.
    """

    def __init__(self, nn_data: NNData, is_rotated: bool = False):
        GenericNNOutput.__init__(self, nn_data)

        self.detections = []
        self.is_rotated = is_rotated
        if is_rotated:
            self.angles = []

    def add(self, label: int, confidence: float, bbox: Tuple[float, ...], angle: int = 0) -> None:
        det = ImgDetection()
        det.label = label
        det.confidence = confidence
        det.xmin = bbox[0]
        det.ymin = bbox[1]
        det.xmax = bbox[2]
        det.ymax = bbox[3]
        self.detections.append(det)
        if self.is_rotated:
            self.angles.append(angle)


@dataclass
class SemanticSegmentation(GenericNNOutput):  # In core, extend from NNData
    """
    Semantic segmentation results, with a mask for each class.

    Examples: `DeeplabV3`, `Lanenet`, `road-semgentation-adas-0001`.
    """
    mask: List[np.ndarray]  # 2D np.array for each class

    def __init__(self, nn_data: NNData, mask: List[np.ndarray]):
        super().__init__(nn_data)
        self.mask = mask


@dataclass
class ImgLandmarks(GenericNNOutput):  # In core, extend from NNData
    """
    Landmarks results, with a list of landmarks and pairs of landmarks to draw lines between.

    Examples: `human-pose-estimation-0001`, `openpose2`, `facial-landmarks-68`, `landmarks-regression-retail-0009`.
    """

    def __init__(self,
                 nn_data: NNData,
                 landmarks: List[List[Any]] = None,
                 landmarks_indices: List[List[int]] = None,
                 pairs: List[Tuple[int, int]] = None,
                 colors: List[Tuple[int, int, int]] = None):
        super().__init__(nn_data)
        self.landmarks = landmarks
        self.landmarks_indices = landmarks_indices
        self.pairs = pairs
        self.colors = colors


@dataclass
class InstanceSegmentation(GenericNNOutput):
    """
    Instance segmentation results, with a mask for each instance.
    """
    # TODO: Finish this, add example models
    masks: List[np.ndarray]  # 2D np.array for each instance
    labels: List[int]  # Class label for each instance

    def __init__(self, nn_data: NNData, masks: List[np.ndarray], labels: List[int]):
        raise NotImplementedError('Instance segmentation not yet implemented')
        super().__init__(nn_data)

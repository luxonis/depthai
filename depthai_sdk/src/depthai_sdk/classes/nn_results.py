"""
General (standarized) NN outputs, to be used for higher-level abstractions (eg. automatic visualization of results).
"SDK supported NN models" will have to have standard NN output, so either dai.ImgDetections, or one of the outputs
below. If the latter, model json config will incldue handler.py logic for decoding to the standard NN output.
These will be integrated into depthai-core, bonus points for on-device decoding of some popular models.
"""
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Any, Union, Optional

import depthai as dai
import numpy as np

from depthai_sdk.visualize.bbox import BoundingBox


@dataclass
class Detection:
    # Original ImgDetection
    img_detection: Union[None, dai.ImgDetection, dai.SpatialImgDetection]
    label_str: str
    confidence: float
    color: Tuple[int, int, int]
    bbox: BoundingBox
    angle: Optional[int]
    ts: Optional[timedelta]

    @property
    def top_left(self) -> Tuple[float, float]:
        return self.bbox.top_left()

    @property
    def bottom_right(self) -> Tuple[float, float]:
        return self.bbox.bottom_right()


@dataclass
class TrackingDetection(Detection):
    tracklet: dai.Tracklet
    filtered_2d: BoundingBox
    filtered_3d: dai.Point3f
    speed: Union[float, None]  # m/s

    @property
    def speed_kmph(self) -> float:
        return self.speed * 3.6

    @property
    def speed_mph(self) -> float:
        return self.speed * 2.236936


@dataclass
class TwoStageDetection(Detection):
    nn_data: dai.NNData


class GenericNNOutput:
    """
    Generic NN output, to be used for higher-level abstractions (eg. automatic visualization of results).
    """

    def __init__(self, nn_data: Union[dai.NNData, dai.ImgDetections, dai.SpatialImgDetections]):
        self.nn_data = nn_data

    def getTimestamp(self) -> timedelta:
        return self.nn_data.getTimestamp()

    def getSequenceNum(self) -> int:
        return self.nn_data.getSequenceNum()


@dataclass
class ExtendedImgDetection(dai.ImgDetection):
    angle: int


# First we have Object detection results, which are already standarized with dai.ImgDetections

@dataclass
class Detections(GenericNNOutput):
    """
    Detection results containing bounding boxes, labels and confidences. Optionally can contain rotation angles.
    """

    def __init__(self,
                 nn_data: Union[dai.NNData, dai.ImgDetections, dai.SpatialImgDetections],
                 is_rotated: bool = False):
        GenericNNOutput.__init__(self, nn_data)
        self.detections: List[ExtendedImgDetection] = []
        self.is_rotated = is_rotated


@dataclass
class SemanticSegmentation(GenericNNOutput):  # In core, extend from NNData
    """
    Semantic segmentation results, with a mask for each class.

    Examples: `DeeplabV3`, `Lanenet`, `road-segmentation-adas-0001`.
    """
    mask: List[np.ndarray]  # 2D np.array for each class

    def __init__(self, nn_data: dai.NNData, mask: List[np.ndarray]):
        super().__init__(nn_data)
        self.mask = mask


@dataclass
class ImgLandmarks(GenericNNOutput):  # In core, extend from NNData
    """
    Landmarks results, with a list of landmarks and pairs of landmarks to draw lines between.

    Examples: `human-pose-estimation-0001`, `openpose2`, `facial-landmarks-68`, `landmarks-regression-retail-0009`.
    """

    def __init__(self,
                 nn_data: dai.NNData,
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

    def __init__(self, nn_data: dai.NNData, masks: List[np.ndarray], labels: List[int]):
        raise NotImplementedError('Instance segmentation not yet implemented')

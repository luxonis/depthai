"""
General (standarized) NN outputs, to be used for higher-level abstractions (eg. automatic visualization of results).
"SDK supported NN models" will have to have standard NN output, so either dai.ImgDetections, or one of the outputs
below. If the latter, model json config will incldue handler.py logic for decoding to the standard NN output.
These will be integrated into depthai-core, bonus points for on-device decoding of some popular models.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, TypeVar, Union, Any

import numpy as np


# First we have Object detection results, which are already standarized with dai.ImgDetections

@dataclass
class SemanticSegmentation:  # In core, extend from NNData
    """
    Provides class for each pixel on the frame.
    Examples: DeeplabV3, Lanenet, road-semgentation-adas-0001
    """
    mask: List[np.array] = field(default_factory=list)  # 2D np.array for each class


@dataclass
class ImgLandmarks:  # In core, extend from NNData
    """
    Provides location of a landmark, eg. joint landmarks, face landmarks, hand landmarks
    Examples: human-pose-estimation-0001, openpose2, facial-landmarks-68, landmarks-regression-retail-0009
    """
    landmarks: List[List[Any]] = field(default_factory=list)
    pairs: List[Tuple[int, int]] = None  # Pairs of landmarks, to draw lines between them
    colors: List[Tuple[int, int, int]] = None  # Color for each landmark (eg. both elbows are in the same color)


@dataclass
class InstanceSegmentations:
    masks: List[np.array] = field(default_factory=list)  # 2D np.array for each instance
    labels: List[int] = field(default_factory=list)  # Class label for each instance


@dataclass
class InstanceSegmentation:
    pass

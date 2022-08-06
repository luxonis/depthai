"""
General (standarized) NN outputs, to be used for higher-level abstractions (eg. automatic visualization of results).
"SDK supported NN models" will have to have standard NN output, so either dai.ImgDetections, or one of the outputs
below. If the latter, model json config will incldue handler.py logic for decoding to the standard NN output.
These will be integrated into depthai-core, bonus points for on-device decoding of some popular models.
"""

import depthai as dai
import numpy as np
from typing import List, Tuple

# First we have Object detection results, which are already standarized with dai.ImgDetections

class SemanticSegmentation: # In core, extend from NNData
    """
    Provides class for each pixel on the frame.
    Examples: DeeplabV3, Lanenet, road-semgentation-adas-0001
    """
    layers: List[np.array] = [] # 2D np.array for each class
    

class ImgLandmarks: # In core, extend from NNData
    """
    Provides location of a landmark, eg. joint landmarks, face landmarks, hand landmarks
    Examples: human-pose-estimation-0001, openpose2, facial-landmarks-68, landmarks-regression-retail-0009
    """
    landmarks: List[dai.Point2f] = [] # Landmarks
    pairs: List[Tuple[int,int]] = None # Pairs of landmarks, to draw lines betwee them
    colors: List[Tuple[int,int,int]] = None # Color for each landmark (eg. both elbows are in the same color)

class InstanceSegmentations(dai.NNData):
    def __init__(self, nnData: dai.NNData) -> None:
        super(nnData).__init__()

class InstanceSegmentation(dai.ImgDetection):
    def __init__(self) -> None:
        super().__init__()


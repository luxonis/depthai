import numpy as np
from depthai import NNData

from depthai_sdk.classes import SemanticSegmentation

NN_WIDTH, NN_HEIGHT = 513, 513


def decode(nn_data: NNData) -> SemanticSegmentation:
    mask = np.array(nn_data.getFirstLayerInt32()).reshape(NN_WIDTH, NN_HEIGHT)
    return SemanticSegmentation(nn_data, mask)

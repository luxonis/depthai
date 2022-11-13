import numpy as np

from depthai_sdk import SemanticSegmentation

NN_WIDTH, NN_HEIGHT = 513, 513


def decode(nn_data) -> SemanticSegmentation:
    mask = np.array(nn_data.getFirstLayerInt32()).reshape(NN_WIDTH, NN_HEIGHT)
    return SemanticSegmentation(mask)

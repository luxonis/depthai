import numpy as np
import depthai as dai
from depthai_sdk.classes.nn_results import ImgLandmarks

THRESHOLD = 0.5

def decode(data: dai.NNData) -> ImgLandmarks:
    # TODO: Use standarized recognition model
    score = np.array(data.getLayerFp16('conv2d_31'))
    score = 1 / (1 + np.exp(-score[0]))  # sigmoid on score
    if score < THRESHOLD:
        return ImgLandmarks(data)

    ldms = np.array(data.getLayerFp16('conv2d_210')).reshape((468, 3))
    landmarks = []
    colors = []
    for ldm in ldms:
        colors.append((0, 0, int(ldm[2]) * 5 + 100))
        landmarks.append((ldm[0] / 192, ldm[1] / 192))
    return ImgLandmarks(data, landmarks=landmarks, colors=colors)

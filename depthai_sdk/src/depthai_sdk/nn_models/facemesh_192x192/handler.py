import numpy as np
import depthai as dai
from typing import Tuple
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

    all_colors = np.array([ldm[2] for ldm in ldms])
    colors = generate_colors_from_z(all_colors)

    for ldm in ldms:
        # colors.append(calculate_color(ldm[2], minz, maxz))
        # colors.append((0, 0, int(ldm[2]) * 5 + 100))
        landmarks.append((ldm[0] / 192, ldm[1] / 192))
    return ImgLandmarks(data, landmarks=landmarks, colors=colors)

def generate_colors_from_z(z_values):
    """
    Generates BGR colors based on normalized Z-values.

    Parameters:
    z_values (numpy.array): Array of Z-values.

    Returns:
    List[Tuple]: List of BGR color tuples.
    """
    def normalize_z_values(z_values, minZ, maxZ):
        return (z_values - minZ) / (maxZ - minZ)

    def map_to_color(normalized_z_values):
        return [(255 - int((1 - value) * 255), 0, 255 - int(value * 255)) for value in normalized_z_values]
    minZ = min(z_values)
    maxZ = max(z_values)
    normalized_z_values = normalize_z_values(z_values, minZ, maxZ)
    return map_to_color(normalized_z_values)
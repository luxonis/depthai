import numpy as np


def sharpness(frame: np.ndarray):
    dx = np.diff(frame)[1:, :]  # remove the first row
    dy = np.diff(frame, axis=0)[:, 1:]  # remove the first column
    dnorm = np.sqrt(dx ** 2 + dy ** 2)
    sharpness = np.average(dnorm)
    return sharpness

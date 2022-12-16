import numpy as np
import depthai as dai

def decode(data: dai.NNData) -> np.ndarray:
    # TODO: Use standarized recognition model
    return data.getData()

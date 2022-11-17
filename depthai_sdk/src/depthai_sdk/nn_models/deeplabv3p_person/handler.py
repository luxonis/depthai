import cv2
import numpy as np

from depthai_sdk import toTensorResult, Previews


def decode(nnManager, packet):
    data = np.squeeze(toTensorResult(packet)["Output/Transpose"])
    classColors = [[0,0,0],  [0,255,0]]
    classColors = np.asarray(classColors, dtype=np.uint8)

    outputColors = np.take(classColors, data, axis=0)
    return outputColors


def draw(nnManager, data, frames):
    if len(data) == 0:
        return

    for name, frame in frames:
        if name == "color" and nnManager.source == "color" and not nnManager._fullFov:
            scaleFactor = frame.shape[0] / nnManager.inputSize[1]
            resizeW = int(nnManager.inputSize[0] * scaleFactor)
            resized = cv2.resize(data, (resizeW, frame.shape[0])).astype(data.dtype)
            offsetW = int(frame.shape[1] - nnManager.inputSize[0] * scaleFactor) // 2
            tailW = frame.shape[1] - offsetW - resizeW
            stacked = np.hstack((np.zeros((frame.shape[0], offsetW, 3)).astype(resized.dtype), resized, np.zeros((frame.shape[0], tailW, 3)).astype(resized.dtype)))
            cv2.addWeighted(frame, 1, stacked, 0.2, 0, frame)
        elif name in (Previews.color.name, Previews.nnInput.name, "host"):
            cv2.addWeighted(frame, 1, cv2.resize(data, frame.shape[:2][::-1]), 0.2, 0, frame)

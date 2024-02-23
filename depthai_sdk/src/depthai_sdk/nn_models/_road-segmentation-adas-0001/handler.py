import cv2
import numpy as np

from depthai_sdk import Previews, toTensorResult


def decode(nnManager, packet):
    # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in packet.getAllLayers()]
    # after squeeze the data.shape is 4,512, 896
    data = np.squeeze(toTensorResult(packet)["L0317_ReWeight_SoftMax"])
    classColors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    classColors = np.asarray(classColors, dtype=np.uint8)

    indices = np.argmax(data, axis=0)
    outputColors = np.take(classColors, indices, axis=0)
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

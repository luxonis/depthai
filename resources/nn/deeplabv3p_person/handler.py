import cv2
import numpy as np

from depthai_helpers.managers import Previews
from depthai_helpers.utils import to_tensor_result


def decode(nn_manager, packet):
    data = np.squeeze(to_tensor_result(packet)["Output/Transpose"])
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output_colors = np.take(class_colors, data, axis=0)
    return output_colors


def draw(nn_manager, data, frames):
    if len(data) == 0:
        return

    for name, frame in frames:
        if name == "color" and nn_manager.source == "color" and not nn_manager.full_fov:
            scale_factor = frame.shape[0] / nn_manager.input_size[1]
            resize_w = int(nn_manager.input_size[0] * scale_factor)
            resized = cv2.resize(data, (resize_w, frame.shape[0])).astype(data.dtype)
            offset_w = int(frame.shape[1] - nn_manager.input_size[0] * scale_factor) // 2
            tail_w = frame.shape[1] - offset_w - resize_w
            stacked = np.hstack((np.zeros((frame.shape[0], offset_w, 3)).astype(resized.dtype), resized, np.zeros((frame.shape[0], tail_w, 3)).astype(resized.dtype)))
            cv2.addWeighted(frame, 1, stacked, 0.2, 0, frame)
        elif name in (Previews.color.name, Previews.nn_input.name, "host"):
            cv2.addWeighted(frame, 1, cv2.resize(data, frame.shape[:2][::-1]), 0.2, 0, frame)

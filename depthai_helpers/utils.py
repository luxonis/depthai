import importlib
from pathlib import Path

import cv2
import numpy as np
import depthai as dai


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple = None) -> np.ndarray:
    if shape is None:
        return arr.transpose(2, 0, 1)
    return cv2.resize(arr, shape).transpose(2, 0, 1)


def to_tensor_result(packet):
    data = {}
    for tensor in packet.getRaw().tensors:
        if tensor.dataType == dai.TensorInfo.DataType.INT:
            data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(tensor.dims[::-1])
        elif tensor.dataType == dai.TensorInfo.DataType.FP16:
            data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims[::-1])
        elif tensor.dataType == dai.TensorInfo.DataType.I8:
            data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(tensor.dims[::-1])
        else:
            print("Unsupported tensor layer type: {}".format(tensor.dataType))
    return data


# https://stackoverflow.com/questions/20656135/python-deep-merge-dictionary-data#20666342
def merge(source, destination):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path.absolute()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

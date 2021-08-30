import importlib
import sys
from pathlib import Path
import urllib.request

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
            data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.FP16:
            data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.I8:
            data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(tensor.dims)
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


def getDeviceInfo(device_id=None):
        device_infos = dai.Device.getAllAvailableDevices()
        if len(device_infos) == 0:
            raise RuntimeError("No DepthAI device found!")
        else:
            print("Available devices:")
            for i, device_info in enumerate(device_infos):
                print(f"[{i}] {device_info.getMxId()} [{device_info.state.name}]")

            if device_id == "list":
                raise SystemExit(0)
            elif device_id is not None:
                matching_device = next(filter(lambda info: info.getMxId() == device_id, device_infos), None)
                if matching_device is None:
                    raise RuntimeError(f"No DepthAI device found with id matching {device_id} !")
                return matching_device
            elif len(device_infos) == 1:
                return device_infos[0]
            else:
                val = input("Which DepthAI Device you want to use: ")
                try:
                    return device_infos[int(val)]
                except:
                    raise ValueError("Incorrect value supplied: {}".format(val))


def show_progress(curr, max):
    done = int(50 * curr / max)
    sys.stdout.write("\r[{}{}] ".format('=' * done, ' ' * (50-done)) )
    sys.stdout.flush()



def downloadYTVideo(video, output_dir=None):
    def progress_func(stream, chunk, bytes_remaining):
        show_progress(stream.filesize - bytes_remaining, stream.filesize)

    try:
        from pytube import YouTube
    except ImportError as ex:
        raise RuntimeError("Unable to use YouTube video due to the following import error: {}".format(ex))
    path = None
    for _ in range(10):
        try:
            path = YouTube(video, on_progress_callback=progress_func)\
                .streams\
                .order_by('resolution')\
                .desc()\
                .first()\
                .download(output_path=output_dir)
        except urllib.error.HTTPError:
            # TODO remove when this issue is resolved - https://github.com/pytube/pytube/issues/990
            # Often, downloading YT video will fail with 404 exception, but sometimes it's successful
            pass
        else:
            break
    if path is None:
        raise RuntimeError("Unable to download YouTube video. Please try again")
    return path

import importlib
import sys
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import depthai as dai


def cos_dist(a, b):
    """
    Calculates cosine distance - https://en.wikipedia.org/wiki/Cosine_similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def frame_norm(frame, bbox):
    """
    Mapps bounding box coordinates (0..1) to pixel values on frame

    Args:
        frame (numpy.ndarray): Frame to which adjust the bounding box
        bbox (list): list of bounding box points in a form of :code:`[x1, y1, x2, y2, ...]`

    Returns:
        list: Bounding box points mapped to pixel values on frame
    """
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple = None) -> np.ndarray:
    """
    Converts interleaved frame into planar

    Args:
        arr (numpy.ndarray): Interleaved frame
        shape (tuple, optional): If provided, the interleaved frame will be scaled to specified shape before converting into planar

    Returns:
        numpy.ndarray: Planar frame
    """
    if shape is None:
        return arr.transpose(2, 0, 1)
    return cv2.resize(arr, shape).transpose(2, 0, 1)


def to_tensor_result(packet):
    """
    Converts NN packet to dict, with each key being output tensor name and each value being correctly reshaped and converted results array

    Useful as a first step of processing NN results for custom neural networks

    Args:
        packet (depthai.NNData): Packet returned from NN node

    Returns:
        dict: Dict containing prepared output tensors
    """
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


def merge(source:dict, destination:dict):
    """
    Utility function to merge two dictionaries

    .. code-block:: python

        a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
        b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
        print(merge(b, a))
        # { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }

    Args:
        source (dict): first dict to merge
        destination (dict): second dict to merge

    Returns:
        dict: merged dict
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
    """
    Loads module from specified path. Used internally e.g. to load a custom handler file from path

    Args:
        path (pathlib.Path): path to the module to be loaded

    Returns:
        module: loaded module from provided path
    """
    spec = importlib.util.spec_from_file_location(path.stem, str(path.absolute()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def getDeviceInfo(device_id=None):
    """
    Find a correct :obj:`depthai.DeviceInfo` object, either matching provided :code:`device_id` or selected by the user (if multiple devices available)
    Useful for almost every app where there is a possibility of multiple devices being connected simultaneously

    Args:
        device_id (str, optional): Specifies device MX ID, for which the device info will be collected

    Returns:
        depthai.DeviceInfo: Object representing selected device info

    Raises:
        RuntimeError: if no DepthAI device was found or, if :code:`device_id` was specified, no device with matching MX ID was found
        ValueError: if value supplied by the user when choosing the DepthAI device was incorrect
    """
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
    """
    Print progressbar to stdout. Each call to this method will write exactly to the same line, so usually it's used as

    .. code-block:: python

        print("Staring processing")
        while processing:
            show_progress(curr_progress, max_progress)
        print(" done") # prints in the same line as progress bar and adds a new line
        print("Processing finished!")

    Args:
        curr (int): Current position on progress bar
        max (int): Maximum position on progress bar
    """
    done = int(50 * curr / max)
    sys.stdout.write("\r[{}{}] ".format('=' * done, ' ' * (50-done)) )
    sys.stdout.flush()



def downloadYTVideo(video, output_dir=None):
    """
    Downloads a video from YouTube and returns the path to video. Will choose the best resolutuion if possible.

    Args:
        video (str): URL to YouTube video
        output_dir (pathlib.Path, optional): Path to directory where youtube video should be downloaded.

    Returns:
         pathlib.Path: Path to downloaded video file

    Raises:
        RuntimeError: thrown when video download was unsuccessful
    """
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

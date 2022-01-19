import importlib
import sys
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import depthai as dai
import datetime as dt
from heapq import heappop, heappush
import threading


def cosDist(a, b):
    """
    Calculates cosine distance - https://en.wikipedia.org/wiki/Cosine_similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def frameNorm(frame, bbox):
    """
    Mapps bounding box coordinates (0..1) to pixel values on frame

    Args:
        frame (numpy.ndarray): Frame to which adjust the bounding box
        bbox (list): list of bounding box points in a form of :code:`[x1, y1, x2, y2, ...]`

    Returns:
        list: Bounding box points mapped to pixel values on frame
    """
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def toPlanar(arr: np.ndarray, shape: tuple = None) -> np.ndarray:
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


def toTensorResult(packet):
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


def loadModule(path: Path):
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


def getDeviceInfo(deviceId=None):
    """
    Find a correct :obj:`depthai.DeviceInfo` object, either matching provided :code:`deviceId` or selected by the user (if multiple devices available)
    Useful for almost every app where there is a possibility of multiple devices being connected simultaneously

    Args:
        deviceId (str, optional): Specifies device MX ID, for which the device info will be collected

    Returns:
        depthai.DeviceInfo: Object representing selected device info

    Raises:
        RuntimeError: if no DepthAI device was found or, if :code:`deviceId` was specified, no device with matching MX ID was found
        ValueError: if value supplied by the user when choosing the DepthAI device was incorrect
    """
    deviceInfos = dai.Device.getAllAvailableDevices()
    if len(deviceInfos) == 0:
        raise RuntimeError("No DepthAI device found!")
    else:
        print("Available devices:")
        for i, deviceInfo in enumerate(deviceInfos):
            print(f"[{i}] {deviceInfo.getMxId()} [{deviceInfo.state.name}]")

        if deviceId == "list":
            raise SystemExit(0)
        elif deviceId is not None:
            matchingDevice = next(filter(lambda info: info.getMxId() == deviceId, deviceInfos), None)
            if matchingDevice is None:
                raise RuntimeError(f"No DepthAI device found with id matching {deviceId} !")
            return matchingDevice
        elif len(deviceInfos) == 1:
            return deviceInfos[0]
        else:
            val = input("Which DepthAI Device you want to use: ")
            try:
                return deviceInfos[int(val)]
            except:
                raise ValueError("Incorrect value supplied: {}".format(val))


def showProgress(curr, max):
    """
    Print progressbar to stdout. Each call to this method will write exactly to the same line, so usually it's used as

    .. code-block:: python

        print("Staring processing")
        while processing:
            showProgress(currProgress, maxProgress)
        print(" done") # prints in the same line as progress bar and adds a new line
        print("Processing finished!")

    Args:
        curr (int): Current position on progress bar
        max (int): Maximum position on progress bar
    """
    done = int(50 * curr / max)
    sys.stdout.write("\r[{}{}] ".format('=' * done, ' ' * (50-done)) )
    sys.stdout.flush()



def downloadYTVideo(video, outputDir=None):
    """
    Downloads a video from YouTube and returns the path to video. Will choose the best resolutuion if possible.

    Args:
        video (str): URL to YouTube video
        outputDir (pathlib.Path, optional): Path to directory where youtube video should be downloaded.

    Returns:
         pathlib.Path: Path to downloaded video file

    Raises:
        RuntimeError: thrown when video download was unsuccessful
    """
    def progressFunc(stream, chunk, bytesRemaining):
        showProgress(stream.filesize - bytesRemaining, stream.filesize)

    try:
        from pytube import YouTube
    except ImportError as ex:
        raise RuntimeError("Unable to use YouTube video due to the following import error: {}".format(ex))
    path = None
    for _ in range(10):
        try:
            path = YouTube(video, on_progress_callback=progressFunc)\
                .streams\
                .order_by('resolution')\
                .desc()\
                .first()\
                .download(output_path=outputDir)
        except urllib.error.HTTPError:
            # TODO remove when this issue is resolved - https://github.com/pytube/pytube/issues/990
            # Often, downloading YT video will fail with 404 exception, but sometimes it's successful
            pass
        else:
            break
    if path is None:
        raise RuntimeError("Unable to download YouTube video. Please try again")
    return path


def cropToAspectRatio(frame, size):
    """
    Crop the frame to desired aspect ratio and then scales it down to desired size
    Args:
        frame (numpy.ndarray): Source frame that will be cropped
        size (tuple): Desired frame size (width, height)
    Returns:
         numpy.ndarray: Cropped frame
    """
    shape = frame.shape
    h = shape[0]
    w = shape[1]
    currentRatio = w / h
    newRatio = size[0] / size[1]

    # Crop width/height to match the aspect ratio needed by the NN
    if newRatio < currentRatio:  # Crop width
        # Use full height, crop width
        newW = (newRatio/currentRatio) * w
        crop = int((w - newW) / 2)
        return frame[:, crop:w-crop]
    else:  # Crop height
        # Use full width, crop height
        newH = (currentRatio/newRatio) * h
        crop = int((h - newH) / 2)
        return frame[crop:h-crop, :]


def resizeLetterbox(frame, size):
    """
    Transforms the frame to meet the desired size, preserving the aspect ratio and adding black borders (letterboxing)
    Args:
        frame (numpy.ndarray): Source frame that will be resized
        size (tuple): Desired frame size (width, height)
    Returns:
         numpy.ndarray: Resized frame
    """
    border_v = 0
    border_h = 0
    if (size[1] / size[0]) >= (frame.shape[0] / frame.shape[1]):
        border_v = int((((size[1] / size[0]) * frame.shape[1]) - frame.shape[0]) / 2)
    else:
        border_h = int((((size[0] / size[1]) * frame.shape[0]) - frame.shape[1]) / 2)
    frame = cv2.copyMakeBorder(frame, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    return cv2.resize(frame, size)


def createBlankFrame(width, height, rgb_color=(0, 0, 0)):
    """
    Create new image(numpy array) filled with certain color in RGB

    Args:
        width (int): New frame width
        height (int): New frame height
        rgb_color (tuple, Optional): Specify frame fill color in RGB format (default (0,0,0) - black)

    Returns:
         numpy.ndarray: New frame filled with specified color
    """
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

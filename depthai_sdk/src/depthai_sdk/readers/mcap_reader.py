from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from mcap.mcap0.reader import make_reader
from mcap.mcap0.stream_reader import StreamReader
from mcap_ros1.decoder import Decoder

from depthai_sdk.previews import PreviewDecoder
from depthai_sdk.readers.abstract_reader import AbstractReader


class McapReader(AbstractReader):
    """
    Reads all saved streams from .mcap recording.
    Supported ROS messages: Image (depth), CompressedImage (left, right, color, disparity)
    """

    def __init__(self, folder: Path) -> None:
        # Get available topics
        with open(folder, "rb") as file:
            reader = make_reader(file)
            channels = reader.get_summary().channels
            self._topics = [c.topic.split('/')[0] for _, c in channels.items()]

        self._readFrames = dict()

        # Init msg array
        for topic in self._topics:
            self._readFrames[topic] = []

        # Create MCAP decoder
        decoder = Decoder(StreamReader(str(folder)))
        self.msgs = decoder.messages
        # Prepare initial frames so we can get frame size

        self._prepareFrames()

    def read(self) -> Dict[str, np.ndarray]:
        """
        Read and return one frame from each available stream.
        """
        self._prepareFrames()
        return self._returnFrames()

    def _prepareFrames(self):
        """
        Read frames until one of each is buffered. Afterwards, call self._returnFrames()
        to get one of each buffered frame.
        """
        while not self._framesReady():
            topic, _, msg = next(self.msgs)
            name = topic.split('/')[0]
            self._readFrames[name].append(self._getCvFrame(msg, name))

    def _getCvFrame(self, msg, name: str):
        """
        Convert ROS message to cv2 frame (numpy array)
        """
        msgType = str(type(msg))
        data = np.frombuffer(msg.data, dtype=np.int8)
        if 'CompressedImage' in msgType:
            if name == 'color':
                return PreviewDecoder.jpegDecode(data, cv2.IMREAD_COLOR)
            else:  # left, right, disparity
                return PreviewDecoder.jpegDecode(data, cv2.IMREAD_GRAYSCALE)
        elif 'Image' in msgType:
            if msg.encoding == 'mono16':
                data = data.view(np.int16)
            return data.reshape((msg.height, msg.width))
            # msg.encoding
        else:
            raise Exception('Only CompressedImage and Image ROS messages are currently supported.')

    def _framesReady(self):
        """
        Check if there is at least one frame from each available stream.
        """
        for _, arr in self._readFrames.items():
            if len(arr) < 1: return False
        return True

    def _returnFrames(self) -> Dict[str, np.ndarray]:
        """
        Return synced frames (one of each available).
        """
        ret = dict()
        for name, arr in self._readFrames.items():
            ret[name] = arr.pop(0)
        return ret

    def getStreams(self) -> List[str]:
        """
        Get available topics
        """
        return [name for name in self._topics]

    def getShape(self, name: str) -> Tuple[int, int]:
        frame = self._readFrames[name][0]
        return (frame.shape[1], frame.shape[0])

    def get_message_size(self, name: str) -> int:
        size = 1
        for shape in self._readFrames[name][0].shape:
            size *= shape
        return size

    def close(self):
        pass

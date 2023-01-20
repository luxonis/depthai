from pathlib import Path
from typing import Any, Generator, List, Dict, Tuple

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

from depthai_sdk.previews import PreviewDecoder
from depthai_sdk.readers.abstract_reader import AbstractReader


class Db3Reader(AbstractReader):
    STREAMS = ['left', 'right', 'depth']

    def __init__(self, folder: Path) -> None:
        self.reader = Reader(str(folder))
        self.reader.open()
        self.generators: Dict[str, Generator] = {}
        self.frames = None  # For shapes

        for con in self.reader.connections:
            for stream in self.STREAMS:
                if stream.lower() in con.topic.lower():
                    self.generators[stream.lower()] = self.reader.messages([con])

        # For shapes
        self.frames = self.read()

        for con in self.reader.connections:
            for stream in self.STREAMS:
                if stream.lower() in con.topic.lower():
                    self.generators[stream.lower()] = self.reader.messages([con])

    def read(self):
        ros_msgs: Dict[str, Any] = dict()
        try:
            for name, gen in self.generators.items():
                con, ts, raw = next(gen)
                ros_msgs[name] = self._getCvFrame(deserialize_cdr(raw, con.msgtype), name)
            return ros_msgs
        except:
            return None

    def disableStream(self, name: str):
        if name in self.generators:
            del self.generators[name]

    def _getCvFrame(self, msg, name: str):
        """
        Convert ROS message to cv2 frame (numpy array)
        """
        msg_type = str(type(msg))
        data = np.frombuffer(msg.data, dtype=np.int8)
        if 'CompressedImage' in msg_type:
            if name == 'color':
                return PreviewDecoder.jpegDecode(data, cv2.IMREAD_COLOR)
            else:  # left, right, disparity
                return PreviewDecoder.jpegDecode(data, cv2.IMREAD_GRAYSCALE)
        elif 'Image' in msg_type:
            if msg.encoding == 'mono16':
                data = data.view(np.int16)
            return data.reshape((msg.height, msg.width))
            # msg.encoding
        else:
            raise Exception('Only CompressedImage and Image ROS messages are currently supported.')

    def getStreams(self) -> List[str]:
        streams = [name for name in self.frames]
        return streams

    def getShape(self, name: str) -> Tuple[int, int]:
        frame = self.frames[name]
        return (frame.shape[1], frame.shape[0])

    def close(self):
        self.reader.close()

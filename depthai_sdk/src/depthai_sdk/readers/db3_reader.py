import array
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.interfaces import Connection
from typing import Any, Generator, List, Dict
import numpy as np
from ..previews import PreviewDecoder
import cv2
from pathlib import Path
import os
from .abstract_reader import AbstractReader

class Db3Reader(AbstractReader):
    STREAMS = ['left', 'right', 'depth']
    generators: Dict[str, Generator] = {}
    frames = None # For shapes
    """
    TODO: make the stream selectable, add function that returns all available streams
    """
    def __init__(self, folder: Path) -> None:
        self.reader = Reader(self._fileWithExt(folder, '.db3'))
        self.reader.open()

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
        rosMsgs: Dict[str, Any] = dict()
        try:
            for name, gen in self.generators.items():
                con, ts, raw = next(gen)
                rosMsgs[name] = self._getCvFrame(deserialize_cdr(raw, con.msgtype), name)
            return rosMsgs
        except:
            return None


    def _getCvFrame(self, msg, name: str):
        """
        Convert ROS message to cv2 frame (numpy array)
        """
        msgType = str(type(msg))
        data = np.frombuffer(msg.data, dtype=np.int8)
        if 'CompressedImage' in msgType:
            if name == 'color':
                return PreviewDecoder.jpegDecode(data, cv2.IMREAD_COLOR)
            else: # left, right, disparity 
                return PreviewDecoder.jpegDecode(data, cv2.IMREAD_GRAYSCALE)
        elif 'Image' in msgType:
            if msg.encoding == 'mono16':
                data = data.view(np.int16)
            return data.reshape((msg.height, msg.width))
            # msg.encoding
        else:
            raise Exception('Only CompressedImage and Image ROS messages are currently supported.')

    def getStreams(self) -> List[str]:
        streams = [name for name in self.frames]
        return streams

    def getShape(self, name: str) -> tuple:
        frame = self.frames[name]
        return (frame.shape[1], frame.shape[0])

    def close(self):
        self.reader.close()
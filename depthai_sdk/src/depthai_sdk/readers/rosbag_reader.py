import array
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import numpy as np

from .abstract_reader import AbstractReader

class RosbagReader(AbstractReader):
    """
    TODO: make the stream selectable, add function that returns all available streams
    """
    def __init__(self, source: str) -> None:
        self.reader = Reader(source)
        self.reader.open()
        if '/device_0/sensor_0/Depth_0/image/data' not in self.reader.topics:
            raise Exception("Provided rosbag can't find required topic (`/device_0/sensor_0/Depth_0/image/data`)")
        self.generator = self.reader.messages('/device_0/sensor_0/Depth_0/image/data')

    def read(self):
        connection, _, rawdata = next(self.generator)
        msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        return msg.data.view(np.int16).reshape((msg.height, msg.width))

    def getStreams(self) -> array:
        return ["depth"] # Only depth recording is supported

    def getShape(self, name: str) -> tuple:
        connection, _, rawdata = next(self.reader.messages('/device_0/sensor_0/Depth_0/image/data'))
        msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        return (msg.width,msg.height)

    def close(self):
        self.reader.close()
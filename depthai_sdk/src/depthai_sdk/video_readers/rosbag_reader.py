from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import numpy as np

from .abstract_reader import AbstractReader

class RosbagReader(AbstractReader):
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

    def getShape(self) -> tuple:
        connection, _, rawdata = next(self.reader.messages('/device_0/sensor_0/Depth_0/image/data'))
        msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        return (msg.width,msg.height)

    def close(self):
        self.reader.close()
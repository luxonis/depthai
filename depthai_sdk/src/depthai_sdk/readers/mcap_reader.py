from mcap.mcap0.stream_reader import StreamReader
from mcap_ros1.decoder import Decoder

from .abstract_reader import AbstractReader

class McapReader(AbstractReader):
    """
    TODO: make the stream selectable, add function that returns all available streams
    """
    i = 0
    def __init__(self, source: str) -> None:
        decoder = Decoder(StreamReader(str(source)))
        self.msgs = decoder.messages

    def read(self):
        topic, record, msg = next(self.msgs)
        if self.i < 20:
            print(topic, record, msg)
        self.i += 1

    def getShape(self) -> tuple:
        # connection, _, rawdata = next(self.reader.messages('/device_0/sensor_0/Depth_0/image/data'))
        # msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        # return (msg.width,msg.height)
        return (0,0)

    def close(self):
        pass
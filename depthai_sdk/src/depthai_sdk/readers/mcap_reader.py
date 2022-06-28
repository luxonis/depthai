from mcap.mcap0.stream_reader import StreamReader

from .abstract_reader import AbstractReader

class McapReader(AbstractReader):
    """
    TODO: make the stream selectable, add function that returns all available streams
    """
    def __init__(self, source: str) -> None:
        stream = open(source, "rb")
        self.reader = StreamReader(stream)

    def read(self):
        record = next(self.reader.records)
        print(type(record))
        # if len(record) < 1000:
            # print(record)

    def getShape(self) -> tuple:
        # connection, _, rawdata = next(self.reader.messages('/device_0/sensor_0/Depth_0/image/data'))
        # msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        # return (msg.width,msg.height)
        return (0,0)

    def close(self):
        self.reader.close()
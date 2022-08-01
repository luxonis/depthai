from depthai_sdk import Replay
import unittest

class TestUtils(unittest.TestCase):

    def test_depthai_recording1(self):
        r = Replay('depth-people-counting-01')
        streams = r.getStreams()
        print(streams)

    def test_depthai_youtube(self):
        r = Replay('depth-people-counting-01')
        streams = r.getStreams()
        print(streams)



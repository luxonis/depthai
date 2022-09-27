#!/usr/bin/env python3
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List

import depthai as dai
from enum import IntEnum

from .classes.packets import FramePacket
from .recorders.abstract_recorder import Recorder
from .oak_outputs.xout import XoutSeqSync, XoutFrames, XoutH26x, XoutMjpeg


def _run(recorders, frameQ: Queue):
    """
    Start recording infinite loop
    """
    while True:
        try:
            frames = frameQ.get()
            if frames is None: # Terminate app
                break
            for name in frames:
                # Save all synced frames into files
                recorders[name].write(name, frames[name])
        except KeyboardInterrupt:
            break
    # Close all recorders - Can't use ExitStack with VideoWriter
    for n in recorders:
        recorders[n].close()
    print('Exiting store frame thread')

class RecordType(IntEnum):
    RAW = 1 # Save raw bitstream
    MP4 = 2 # Containerize into mp4 file, requires `av` library
    MCAP = 3 # To .mcap
    BAG = 4 # To ROS .bag

class Codec(IntEnum):
    NONE = 0
    MJPEG = 1
    H264 = 2
    H265 = 3
    @classmethod
    def fourcc(cls, codec: 'Codec') -> str:
        if codec == cls.MJPEG:
            return 'mjpeg'
        elif codec == cls.H264:
            return 'h264'
        elif codec == cls.H265:
            return 'hevc'

class Record(XoutSeqSync):
    """
    This class records depthai streams from OAK cameras into different formats.
    Available formats: .h265, .mjpeg, .mp4, .mcap, .bag
    It will also save calibration .json, so depth reconstruction will 
    """

    def package(self, msgs: Dict):
        # Here we get sequence-num synced messages:)
        pass

    def visualize(self, packet: FramePacket) -> None:
        pass # No need.


    def __init__(self, path: Path, type: RecordType):
        """
        Args:
            path (Path): Path to the recording folder
            device (dai.Device): OAK device object
        """
        super().__init__([]) # We don't yet have streams, we will set it up later
        self.folder = path
        self.type = type

    def start(self, device: dai.Device, xouts: List[XoutFrames]):
        """
        Start recording process. This will create and start the pipeline,
        start recording threads, and initialize all queues.
        """
        self.streams = [out.frames for out in xouts] # required by XoutSeqSync

        self.mxid = device.getMxId()
        self.path = self._createFolder(self.folder, self.mxid)

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

        self.frame_q = Queue(maxsize=20)
        self.process = Thread(target=_run, args=(self._getRecorders(device, xouts), self.frame_q))
        self.process.start()

    def _getRecorders(self, device: dai.Device, xouts: List[XoutFrames]) -> Dict[str, Recorder]:
        """
        Create recorders
        """
        recorders = dict()

        codecs: Dict[str, Codec] = {}
        for xout in xouts:
            if isinstance(xout, XoutH26x):
            codecs[xout.frames.name]

        if self.type == RecordType.MCAP:
            from .recorders.mcap_recorder import McapRecorder
            rec = McapRecorder(self.path, device)
            for xout in xouts:
                if isinstance(xout, XoutH26x):
                    raise Exception("MCAP recording only supports MJPEG encoding!")
                if isinstance(xout, XoutMjpeg) and xout.lossless:
                    # Foxglove Studio doesn't support Lossless MJPEG
                    raise Exception("MCAP recording doesn't support Lossless MJPEG encoding!")
                # rec.setPointcloud(self._pointcloud)
                recorders[xout.frames.name] = rec
        if self.type == RecordType.MP4:
            from .recorders.pyav_mp4_recorder import PyAvRecorder
            rec = PyAvRecorder(self.path, )

        # if 'depth' in save:
        #     from .recorders.rosbag_recorder import RosbagRecorder
        #     recorders['depth'] = RosbagRecorder(self.path, device, self.getSizes())
        #     save.remove('depth')
        #
        # if len(save) == 0: return recorders

        # else:
        #     try:
        #         # Try importing av
        #         from .recorders.pyav_mp4_recorder import PyAvRecorder
        #         rec = PyAvRecorder(self.path, self.quality, self.args.rgbFps, self.args.monoFps)
        #     except:
        #         print("'av' library is not installed, depthai-record will save raw encoded streams.")
        #         from .recorders.raw_recorder import RawRecorder
        #         rec = RawRecorder(self.path, self.quality)
        # # All other streams ("color", "left", "right", "disparity") will use
        # # the same Raw/PyAv recorder
        # for name in save:
        #     recorders[name] = rec
        return recorders


    def _createFolder(self, path: Path, mxid: str) -> Path:
        """
        Creates recording folder
        """
        i = 0
        while True:
            i += 1
            recordings_path = path / f"{i}-{str(mxid)}"
            if not recordings_path.is_dir():
                recordings_path.mkdir(parents=True, exist_ok=False)
                return recordings_path

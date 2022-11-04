#!/usr/bin/env python3
from enum import IntEnum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List

import depthai as dai

from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.oak_outputs.xout import XoutSeqSync, XoutFrames
from depthai_sdk.recorders.abstract_recorder import Recorder


def _run(recorder, frameQ: Queue):
    """
    Start recording infinite loop
    """
    while True:
        try:
            frames = frameQ.get()
            if frames is None:  # Terminate app
                break

            for name in frames:
                # Save all synced frames into files
                recorder.write(name, frames[name])
        except KeyboardInterrupt:
            break
    # Close all recorders - Can't use ExitStack with VideoWriter
    recorder.close()
    print('Exiting store frame thread')


class RecordType(IntEnum):
    VIDEO = 1  # Save to video file
    BAG = 2  # To ROS .bag
    # MCAP = 3 # To .mcap


class Record(XoutSeqSync):
    """
    This class records depthai streams from OAK cameras into different formats.
    Available formats: .h265, .mjpeg, .mp4, .mcap, .bag
    It will also save calibration .json, so depth reconstruction will
    """
    name_mapping: Dict[str, str]  # XLinkOut stream name -> Friendly name mapping

    def package(self, msgs: Dict):
        # Here we get sequence-num synced messages:)
        print('package new frames record', msgs)
        mapped = dict()
        for name, msg in msgs.items():
            if name in self.name_mapping:  # Map to friendly name
                mapped[self.name_mapping[name]] = msg
            else:
                mapped[name] = msg

        self.frame_q.put(mapped)

    def visualize(self, packet: FramePacket) -> None:
        pass  # No need.

    def __init__(self,
                 path: Path,
                 type: RecordType):
        """
        Args:
            path (Path): Path to the recording folder
            device (dai.Device): OAK device object
        """
        super().__init__([])  # We don't yet have streams, we will set it up later
        self.folder = path
        self.type = type

    def start(self, device: dai.Device, xouts: List[XoutFrames]):
        """
        Start recording process. This will create and start the pipeline,
        start recording threads, and initialize all queues.
        """
        self._streams = [out.frames.name for out in xouts]  # required by XoutSeqSync
        self.streamNum = len(self._streams)

        self.name_mapping = dict()
        for xout in xouts:
            if xout.frames.friendly_name:
                self.name_mapping[xout.frames.name] = xout.frames.friendly_name

        self.mxid = device.getMxId()
        self.path = self._createFolder(self.folder, self.mxid)

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

        self.frame_q = Queue(maxsize=20)

        self.process = Thread(target=_run, args=(self._get_recorder(device, xouts), self.frame_q))
        self.process.start()

    def _get_recorder(self, device: dai.Device, xouts: List[XoutFrames]) -> Recorder:
        """
        Create recorder
        """
        # if self.type == RecordType.MCAP:
        #     from .recorders.mcap_recorder import McapRecorder
        #     return McapRecorder(self.path, device, xouts)
        if self.type == RecordType.VIDEO:
            from .recorders.video_recorder import VideoRecorder
            return VideoRecorder(self.path, xouts)
        elif self.type == RecordType.BAG:
            from .recorders.rosbag_recorder import RosbagRecorder
            return RosbagRecorder(self.path, device, )
        else:
            raise ValueError(f"Recording type '{self.type}' isn't supported!")

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

    def close(self):
        self.frame_q.put(None)  # Close recorder and stop the thread

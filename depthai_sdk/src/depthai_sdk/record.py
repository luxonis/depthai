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


def _run(recorder: Recorder, frameQ: Queue):
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
    MCAP = 3  # To .mcap


class Record(XoutSeqSync):
    """
    This class records depthai streams from OAK cameras into different formats.
    Available formats: .h265, .mjpeg, .mp4, .mcap, .bag
    It will also save calibration .json, so depth reconstruction will
    """
    name_mapping: Dict[str, str]  # XLinkOut stream name -> Friendly name mapping
    frame_q: Queue = None
    recorder: Recorder

    def package(self, msgs: Dict):
        # Here we get sequence-num synced messages:)
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

        if self.type == RecordType.MCAP:
            from .recorders.mcap_recorder import McapRecorder
            self.recorder = McapRecorder()
        elif self.type == RecordType.VIDEO:
            from .recorders.video_recorder import VideoRecorder
            self.recorder = VideoRecorder()
        elif self.type == RecordType.BAG:
            from .recorders.rosbag_recorder import RosbagRecorder
            self.recorder = RosbagRecorder()
        else:
            raise ValueError(f"Recording type '{self.type}' isn't supported!")

    def no_sync(self, name: str, msg):
        # name = self.name_mapping[name] if name in self.name_mapping else name
        obj = {name: msg}
        self.frame_q.put(obj)

    def start(self, device: dai.Device, xouts: List[XoutFrames]):
        """
        Start recording process. This will create and start the pipeline,
        start recording threads, and initialize all queues.
        """
        if self.type == RecordType.VIDEO:
            self._streams = [out.frames.name for out in xouts]  # required by XoutSeqSync
            self.streamNum = len(xouts)
            self.name_mapping = dict()
            for xout in xouts:
                self.name_mapping[xout.frames.name] = xout.name
        else: # For MCAP/Ros bags we don't need msg syncing
            self.newMsg = self.no_sync

        self.mxid = device.getMxId()
        self.path = self._createFolder(self.folder, self.mxid)
        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

        self.recorder.update(self.path, device, xouts)

        self.frame_q = Queue(maxsize=20)
        self.process = Thread(target=_run, args=(self.recorder, self.frame_q))
        self.process.start()

    # TODO: support pointclouds in MCAP
    def config_mcap(self, pointcloud: bool):
        if self.type != RecordType.MCAP:
            print(f"Recorder type is {self.type}, not MCAP! Config attempt ignored.")
            return
        self.recorder.setPointcloud(pointcloud)

    # def config_video(self, ):
    # Nothing to configure for video recorder

    # TODO: implement config of BAG to either record depth as frame or pointcloud
    # def config_bag(self, pointcloud: bool):
    #     if self.type != RecordType.BAG:
    #         print(f"Recorder type is {self.type}, not BAG! Config attempt ignored.")
    #     self.recorder.set_pointcloud(pointcloud)


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
        if self.frame_q:
            self.frame_q.put(None)  # Close recorder and stop the thread

#!/usr/bin/env python3
from enum import IntEnum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List

import depthai as dai

from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_seq_sync import XoutSeqSync
from depthai_sdk.recorders.abstract_recorder import Recorder


def _run(recorder: Recorder, frame_queue: Queue):
    """
    Start recording infinite loop
    """
    while True:
        try:
            frames = frame_queue.get()
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

    def __init__(self, path: Path, record_type: RecordType):
        """
        Args:
            path (Path): Path to the recording folder
            record_type (RecordType): Recording type
        """
        super().__init__([])  # We don't yet have streams, we will set it up later
        self.folder = path
        self.record_type = record_type
        self.frame_q = None
        self.name_mapping = None  # XLinkOut stream name -> Friendly name mapping

        self.stream_num = None
        self.mxid = None
        self.path = None
        self.process = None

        if self.record_type == RecordType.MCAP:
            from .recorders.mcap_recorder import McapRecorder
            self.recorder = McapRecorder()
        elif self.record_type == RecordType.VIDEO:
            from .recorders.video_recorder import VideoRecorder
            self.recorder = VideoRecorder()
        elif self.record_type == RecordType.BAG:
            from .recorders.rosbag_recorder import RosbagRecorder
            self.recorder = RosbagRecorder()
        else:
            raise ValueError(f"Recording type '{self.record_type}' isn't supported!")

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

    def no_sync(self, name: str, msg):
        # name = self.name_mapping[name] if name in self.name_mapping else name
        obj = {name: msg}
        self.frame_q.put(obj)

    def start(self, device: dai.Device, xouts: List[XoutFrames]):
        """
        Start recording process. This will create and start the pipeline,
        start recording threads, and initialize all queues.
        """
        if self.record_type == RecordType.VIDEO:
            self._streams = [out.frames.name for out in xouts]  # required by XoutSeqSync
            self.stream_num = len(xouts)
            self.name_mapping = dict()
            for xout in xouts:
                self.name_mapping[xout.frames.name] = xout.name
        else:  # For MCAP/Ros bags we don't need msg syncing
            self.new_msg = self.no_sync

        self.mxid = device.getMxId()
        self.path = self._create_folder(self.folder, self.mxid)
        calib_data = device.readCalibration()
        calib_data.eepromToJsonFile(str(self.path / "calib.json"))

        self.recorder.update(self.path, device, xouts)

        self.frame_q = Queue(maxsize=20)
        self.process = Thread(target=_run, args=(self.recorder, self.frame_q))
        self.process.start()

    # TODO: support pointclouds in MCAP
    def config_mcap(self, pointcloud: bool):
        if self.record_type != RecordType.MCAP:
            print(f"Recorder type is {self.record_type}, not MCAP! Config attempt ignored.")
            return
        self.recorder.set_pointcloud(pointcloud)

    # def config_video(self, ):
    # Nothing to configure for video recorder

    # TODO: implement config of BAG to either record depth as frame or pointcloud
    # def config_bag(self, pointcloud: bool):
    #     if self.type != RecordType.BAG:
    #         print(f"Recorder type is {self.type}, not BAG! Config attempt ignored.")
    #     self.recorder.set_pointcloud(pointcloud)

    def _create_folder(self, path: Path, mxid: str) -> Path:
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

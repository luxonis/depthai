#!/usr/bin/env python3
from enum import IntEnum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List

import depthai as dai

from depthai_sdk.classes.packets import FramePacket, IMUPacket
from depthai_sdk.logger import LOGGER
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
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
    LOGGER.info('Exiting store frame thread')


class RecordType(IntEnum):
    VIDEO = 1  # Save to video file
    VIDEO_LOSSLESS = 2  # Save to lossless video file (.avi)
    ROSBAG = 3  # To ROS .bag
    MCAP = 4  # To .mcap
    DB3 = 5  # To .db3 (ros2)


class Record:
    """
    This class records depthai streams from OAK cameras into different formats.
    It will also save calibration .json, so depth reconstruction will be possible.
    """

    def __init__(self, path: Path, record_type: RecordType):
        """
        Args:
            path (Path): Path to the recording folder
            record_type (RecordType): Recording type
        """
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
        elif self.record_type == RecordType.VIDEO_LOSSLESS:
            from .recorders.video_recorder import VideoRecorder
            self.recorder = VideoRecorder(lossless=True)
        elif self.record_type == RecordType.ROSBAG:
            from .recorders.rosbag_recorder import Rosbag1Recorder
            self.recorder = Rosbag1Recorder()
        elif self.record_type == RecordType.DB3:
            from .recorders.rosbag_recorder import Rosbag2Recorder
            self.recorder = Rosbag2Recorder()
        else:
            raise ValueError(f"Recording type '{self.record_type}' isn't supported!")

    def write(self, packets):
        if not isinstance(packets, dict):
            packets = {packets.name: packets}

        msgs = dict()
        for name, packet in packets.items():
            if isinstance(packet, FramePacket):
                msgs[name] = packet.msg
            elif isinstance(packet, IMUPacket):
                msgs[name] = packet.packet
        self.frame_q.put(msgs)

    def start(self, device: dai.Device, xouts: List[XoutFrames]):
        """
        Start recording process. This will create and start the pipeline,
        start recording threads, and initialize all queues.
        """
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
            LOGGER.info(f"Recorder type is {self.record_type}, not MCAP! Config attempt ignored.")
            return
        self.recorder.set_pointcloud(pointcloud)

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

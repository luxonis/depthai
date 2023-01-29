import time
from datetime import timedelta, datetime
from pathlib import Path
from threading import Thread
from typing import Union, Callable, Dict, List

import depthai as dai

from depthai_sdk import Component, FramePacket
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.trigger_action.actions.abstract_action import Action


class RecordAction(Action):
    def __init__(self,
                 inputs: Union[Component, Callable, List[Union[Component, Callable]]],
                 path: str,
                 duration_before_trigger: Union[int, timedelta],
                 duration_after_trigger: Union[timedelta, int]
                 ):
        super().__init__(inputs)
        self.path = Path(path).resolve()
        if isinstance(duration_before_trigger, timedelta):
            duration_before_trigger = duration_before_trigger.total_seconds()
        if isinstance(duration_after_trigger, timedelta):
            duration_after_trigger = duration_after_trigger.total_seconds()
        if duration_before_trigger > 0 and duration_after_trigger > 0:
            self.duration_bt = duration_before_trigger
            self.duration_at = duration_after_trigger
        else:
            raise ValueError("Recording durations before and after trigger must be positive integers "
                             "or timedelta objects representing positive time difference")
        self.recorder = VideoRecorder()
        self.stream_names = []  # will be assigned during recorder setup
        self.buffers_status: Dict[str, Dict[str, Union[int, List[int]]]] = {}

    def _run(self):
        while True:
            rbt_ids = self.buffers_status['ready']['bt']

            # Wait until there is at least one ready before trigger buffer
            while not len(rbt_ids):
                time.sleep(0.01)
            rbt_id = rbt_ids.pop(0)

            # Create file for every stream
            buf_name = f'bt_{rbt_id}'
            subfolder = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            self.recorder.create_files_for_buffer(subfolder=subfolder, buf_name=buf_name)

            # Write before trigger buffer into file for every stream
            for stream_name in self.stream_names:
                while not self.recorder.is_buffer_empty(stream_name, buf_name):
                    self.recorder.write_from_buffer(wr_name=stream_name,
                                                    buf_name=buf_name,
                                                    n_elems=1)
                    time.sleep(0.01)

            # Wait until after trigger buffer with the same id is ready
            while rbt_id not in self.buffers_status['ready']['at']:
                time.sleep(0.01)

            # Write after trigger buffer into the same file for every stream
            buf_name = f'at_{rbt_id}'
            for stream_name in self.stream_names:
                while not self.recorder.is_buffer_empty(stream_name, buf_name):
                    self.recorder.write_from_buffer(wr_name=stream_name,
                                                    buf_name=buf_name,
                                                    n_elems=1)
                    time.sleep(0.01)

            # Close files
            print(f'Saved to {str(self.path / subfolder)}')
            self.recorder.close_files()

    def after_trigger(self):
        # Setup for the current recording
        wbt_id = self.buffers_status['writing']['bt']
        self.buffers_status['ready']['bt'].append(wbt_id)
        self.buffers_status['writing']['at'].append(wbt_id)

        # Setup for the next recording
        new_id = wbt_id + 1
        buffers = {f'bt_{new_id}': self.duration_bt,
                   f'at_{new_id}': self.duration_at}
        self.recorder.init_buffers(buffers)
        self.buffers_status['writing']['bt'] = new_id

    def process_packets(self, packets: Dict[str, FramePacket]):
        # {'color_video': FramePacket, 'color_bitstream': FramePacket, 'stereo_disparity': DisparityPacket}
        # print(packets)

        # Write into the only writing before trigger buffer
        wbt_id = self.buffers_status['writing']['bt']

        # Extract imgFrames from packets --> Syncing packets is redundant for RecordAction, just sync frames
        frames = dict.fromkeys(self.stream_names)
        for name, _ in frames.items():
            for packet_name, packet in packets.items():
                if name in packet_name:
                    frames[name] = packet.imgFrame
                    packets.pop(packet_name)
                    break

        for _, frame in frames.items():
            if frame is None:
                raise Exception("Extracting imgFrames from packets failed, some naming problem")

        self.recorder.add_to_buffers(f'bt_{wbt_id}', frames)  # does for every stream

        # Write into all writing after trigger buffers
        wat_ids = self.buffers_status['writing']['at']
        for i, buffer_id in enumerate(list(wat_ids)):
            buf_name = f'at_{buffer_id}'
            all_streams_ready = True
            for stream_name in self.stream_names:
                if not self.recorder.is_buffer_full(stream_name, buf_name):
                    self.recorder.add_to_buffer(stream_name, buf_name, frames[stream_name])
                    all_streams_ready = False
            if all_streams_ready:
                wat_ids.pop(i)
                self.buffers_status['ready']['at'].append(buffer_id)

    def setup(self, device: dai.Device, xouts: List['XoutFrames']):
        self.stream_names = [xout.frames.name for xout in xouts]  # e.g., [color_video, color_bitstream]
        print('[xout.frames.name] = ', self.stream_names)
        self.recorder.update(self.path, device, xouts)

    def run_thread(self):
        buffers = {'bt_1': self.duration_bt, 'at_1': self.duration_at}
        self.recorder.init_buffers(buffers)  # does for every stream
        self.buffers_status = {'writing': {'bt': 1, 'at': []},  # there is always only one writing before trigger buffer
                               'ready': {'bt': [], 'at': []}}

        process = Thread(target=self._run)
        process.start()

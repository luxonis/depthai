import time
from abc import abstractmethod, ABC
from collections import deque
from datetime import timedelta, datetime
from pathlib import Path
from threading import Thread
from typing import Callable, List, Union, Dict, Tuple

import depthai as dai

from depthai_sdk import FramePacket
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger


class TriggerActionController(ABC):
    last_trigger_time: datetime = datetime.min
    trigger_condition: Callable
    trigger_period: Union[timedelta, int]

    def __init__(self, trigger: Trigger):
        self.trigger_condition = trigger.condition
        self.trigger_period = trigger.period

    @abstractmethod
    def start(self, device: dai.Device, xout: 'XoutFrames'):  # do we need device here?
        raise NotImplementedError()

    @abstractmethod
    def new_packet_trigger(self, packet: FramePacket, _=None):
        raise NotImplementedError()

    @abstractmethod
    def new_packet_action(self, packet: FramePacket, _=None):
        raise NotImplementedError()


class RecordController(TriggerActionController):
    path: Path
    duration: Tuple[int, int]
    recorder: VideoRecorder
    stream_name: str
    buffers_status: Dict[str, Dict[str, Union[int, List[int]]]]
    last_ts: timedelta = timedelta(0)

    def __init__(self, trigger: Trigger, action: RecordAction):
        super().__init__(trigger)
        self.path = action.path
        self.duration = action.duration
        self.recorder = VideoRecorder()

    def _run(self):
        while True:
            ready_before_ids = self.buffers_status['ready']['before_trigger']

            # Wait until there is at least one ready before_trigger buffer
            while not len(ready_before_ids):
                time.sleep(0.01)
            ready_before_id = ready_before_ids.pop(0)

            # Create cv2.VideoWriter file
            buf_name = f'before_trigger_{ready_before_id}'
            subfolder = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            file = self.recorder.create_file(wr_name=self.stream_name,
                                             subfolder=subfolder,
                                             filename=self.stream_name,
                                             buf_name=buf_name)  # buf_name is needed to get its frames' properties

            # Write before_trigger buffer into file
            while not self.recorder.is_buffer_empty(self.stream_name, buf_name):
                self.recorder.write_to_file(wr_name=self.stream_name,
                                            buf_name=buf_name,
                                            file=file)
                time.sleep(0.01)

            # Wait until after_trigger buffer with the same id is ready
            while ready_before_id not in self.buffers_status['ready']['after_trigger']:
                time.sleep(0.01)

            # Write after_trigger buffer into the same file
            buf_name = f'after_trigger_{ready_before_id}'
            while not self.recorder.is_buffer_empty(self.stream_name, buf_name):
                self.recorder.write_to_file(wr_name=self.stream_name,
                                            buf_name=buf_name,
                                            file=file)
                time.sleep(0.01)

            # Close file
            print(f'Saved to {str(self.path / subfolder)}')
            file.release()

    def new_packet_trigger(self, packet: FramePacket, _=None):  # visualizer seems redundant here
        if self.trigger_condition(packet):
            trigger_time = datetime.now()
            if trigger_time - self.last_trigger_time > self.trigger_period:
                print("TRIGGERED!")
                self.last_trigger_time = trigger_time

                # Setup for the current recording
                writing_before_id = self.buffers_status['writing']['before_trigger']
                self.buffers_status['ready']['before_trigger'].append(writing_before_id)
                self.buffers_status['writing']['after_trigger'].append(writing_before_id)

                # Setup for the next recording
                new_buf_id = writing_before_id + 1
                buffers = {f'before_trigger_{new_buf_id}': self.duration[0],
                           f'after_trigger_{new_buf_id}': self.duration[1]}
                self.recorder.init_buffers(self.stream_name, buffers)
                self.buffers_status['writing']['before_trigger'] = new_buf_id

    def new_packet_action(self, packet: FramePacket, _=None):  # visualizer seems redundant here
        # Write into the only writing before_trigger buffer
        buffer_before_id = self.buffers_status['writing']['before_trigger']
        self.recorder.add_to_buffer(self.stream_name, f'before_trigger_{buffer_before_id}', packet.frame)

        # Write into all writing after_trigger buffers
        buffer_after_ids = self.buffers_status['writing']['after_trigger']
        for i, buffer_id in enumerate(list(buffer_after_ids)):
            buffer = f'after_trigger_{buffer_id}'
            if not self.recorder.is_buffer_full(self.stream_name, buffer):
                self.recorder.add_to_buffer(self.stream_name, buffer, packet.frame)
            else:
                buffer_after_ids.pop(i)
                self.buffers_status['ready']['after_trigger'].append(buffer_id)

    def start(self, device: dai.Device, xout: 'XoutFrames'):  # probably extend to multiple xouts
        self.stream_name = xout.name  # what's the difference between xout.name and xout.frames.name?

        # Check what are the names
        print('xout.name = ', xout.name)
        print('xout.frames.name = ', xout.frames.name)

        self.recorder.update(self.path, device, [xout])  # creates video_writers

        buffers = {'before_trigger_1': self.duration[0], 'after_trigger_1': self.duration[1]}
        self.recorder.init_buffers(self.stream_name, buffers)
        self.buffers_status = {'writing': {'before_trigger': 1, 'after_trigger': []},
                               'ready': {'before_trigger': [], 'after_trigger': []}}

        process = Thread(target=self._run)
        process.start()




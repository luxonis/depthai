import time
from abc import abstractmethod, ABC
from datetime import timedelta, datetime
from threading import Thread
from typing import List, Union, Dict

import depthai as dai

from depthai_sdk import FramePacket
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger


class TriggerActionController(ABC):
    def __init__(self, trigger: Trigger):
        self.last_trigger_time = datetime.min
        self.trigger_condition = trigger.condition
        self.trigger_cooldown = trigger.cooldown

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
    def __init__(self, trigger: Trigger, action: RecordAction):
        super().__init__(trigger)
        self.path = action.path
        self.duration_bt = action.duration_before_trigger
        self.duration_at = action.duration_after_trigger
        self.recorder = VideoRecorder()
        self.stream_name = ''  # will be assigned during recorder setup... TODO: is it OK?
        self.buffers_status: Dict[str, Dict[str, Union[int, List[int]]]] = {}
        self.last_ts = timedelta(0)

    def _run(self):
        while True:
            rbt_ids = self.buffers_status['ready']['bt']

            # Wait until there is at least one ready before trigger buffer
            while not len(rbt_ids):
                time.sleep(0.01)
            rbt_id = rbt_ids.pop(0)

            # Create cv2.VideoWriter file
            buf_name = f'bt_{rbt_id}'
            subfolder = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            self.recorder.create_file(wr_name=self.stream_name,
                                      subfolder=subfolder,
                                      buf_name=buf_name)

            # Write before trigger buffer into file
            while not self.recorder.is_buffer_empty(self.stream_name, buf_name):
                self.recorder.write_from_buffer(wr_name=self.stream_name,
                                                buf_name=buf_name,
                                                n_elems=1)
                time.sleep(0.01)

            # Wait until after trigger buffer with the same id is ready
            while rbt_id not in self.buffers_status['ready']['at']:
                time.sleep(0.01)

            # Write after trigger buffer into the same file
            buf_name = f'at_{rbt_id}'
            while not self.recorder.is_buffer_empty(self.stream_name, buf_name):
                self.recorder.write_from_buffer(wr_name=self.stream_name,
                                                buf_name=buf_name,
                                                n_elems=1)
                time.sleep(0.01)

            # Close file
            print(f'Saved to {str(self.path / subfolder)}')
            self.recorder.close_writer(self.stream_name)

    def new_packet_trigger(self, packet: FramePacket, _=None):  # visualizer seems redundant here
        if self.trigger_condition(packet):
            trigger_time = datetime.now()
            if trigger_time - self.last_trigger_time > self.trigger_cooldown:
                print("TRIGGERED!")
                self.last_trigger_time = trigger_time

                # Setup for the current recording
                wbt_id = self.buffers_status['writing']['bt']
                self.buffers_status['ready']['bt'].append(wbt_id)
                self.buffers_status['writing']['at'].append(wbt_id)

                # Setup for the next recording
                new_id = wbt_id + 1
                buffers = {f'bt_{new_id}': self.duration_bt,
                           f'at_{new_id}': self.duration_at}
                self.recorder.init_buffers(self.stream_name, buffers)
                self.buffers_status['writing']['bt'] = new_id

    def new_packet_action(self, packet: FramePacket, _=None):  # visualizer seems redundant here
        # Write into the only writing before trigger buffer
        wbt_id = self.buffers_status['writing']['bt']
        self.recorder.add_to_buffer(self.stream_name, f'bt_{wbt_id}', packet.frame)

        # Write into all writing after trigger buffers
        wat_ids = self.buffers_status['writing']['at']
        for i, buffer_id in enumerate(list(wat_ids)):
            buffer = f'at_{buffer_id}'
            if not self.recorder.is_buffer_full(self.stream_name, buffer):
                self.recorder.add_to_buffer(self.stream_name, buffer, packet.frame)
            else:
                wat_ids.pop(i)
                self.buffers_status['ready']['at'].append(buffer_id)

    def start(self, device: dai.Device, xout: 'XoutFrames'):  # probably extend to multiple xouts
        self.stream_name = xout.name

        self.recorder.update(self.path, device, [xout])  # creates video_writers

        buffers = {'bt_1': self.duration_bt, 'at_1': self.duration_at}
        self.recorder.init_buffers(self.stream_name, buffers)
        self.buffers_status = {'writing': {'bt': 1, 'at': []},  # there is always only one writing before trigger buffer
                               'ready': {'bt': [], 'at': []}}

        process = Thread(target=self._run)
        process.start()




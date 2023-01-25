from abc import abstractmethod
from pathlib import Path
from typing import Optional, Callable, List

import depthai as dai

from depthai_sdk import FramePacket
from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.controllers import TriggerActionController, RecordController
from depthai_sdk.oak_outputs.syncing import SequenceNumSync
from depthai_sdk.oak_outputs.xout import XoutFrames, XoutDepth
from depthai_sdk.oak_outputs.xout_base import XoutBase
from depthai_sdk.record import Record
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger
from depthai_sdk.visualize import Visualizer


class BaseConfig:
    @abstractmethod
    def setup(self, pipeline: dai.Pipeline, device, names: List[str]) -> List[XoutBase]:
        raise NotImplementedError()


class OutputConfig(BaseConfig):
    """
    Saves callbacks/visualizers until the device is fully initialized. I'll admit it's not the cleanest solution.
    """

    def __init__(self,
                 output: Callable,
                 callback: Callable,
                 visualizer: Visualizer = None,
                 visualizer_enabled: bool = False,
                 record_path: Optional[str] = None):
        self.output = output  # Output of the component (a callback)
        self.callback = callback  # Callback that gets called after syncing
        self.visualizer = visualizer
        self.visualizer_enabled = visualizer_enabled
        self.record_path = record_path

    def find_new_name(self, name: str, names: List[str]):
        while True:
            arr = name.split(' ')
            num = arr[-1]
            if num.isnumeric():
                arr[-1] = str(int(num) + 1)
                name = " ".join(arr)
            else:
                name = f"{name} 2"
            if name not in names:
                return name

    def setup(self, pipeline: dai.Pipeline, device, names: List[str]) -> List[XoutBase]:
        xoutbase: XoutBase = self.output(pipeline, device)
        xoutbase.setup_base(self.callback)

        if xoutbase.name in names:  # Stream name already exist, append a number to it
            xoutbase.name = self.find_new_name(xoutbase.name, names)
        names.append(xoutbase.name)

        recorder = None
        if self.record_path:
            recorder = VideoRecorder()

            if isinstance(xoutbase, XoutDepth):
                raise NotImplementedError('Depth recording is not implemented yet.'
                                          'Please use OakCamera.record() instead.')

            recorder.update(Path(self.record_path), device, [xoutbase])

        if self.visualizer:
            xoutbase.setup_visualize(visualizer=self.visualizer,
                                     visualizer_enabled=self.visualizer_enabled,
                                     name=xoutbase.name)

        if self.record_path:
            xoutbase.setup_recorder(recorder=recorder)

        return [xoutbase]


class RecordConfig(BaseConfig):
    def __init__(self, outputs: List[Callable], rec: Record):
        self.outputs = outputs
        self.rec = rec

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, _) -> List[XoutBase]:
        xouts: List[XoutFrames] = []
        for output in self.outputs:
            xoutbase: XoutFrames = output(pipeline, device)
            xoutbase.setup_base(None)
            xouts.append(xoutbase)

        self.rec.setup_base(None)
        self.rec.start(device, xouts)

        return [self.rec]


class SyncConfig(BaseConfig, SequenceNumSync):
    def __init__(self, outputs: List[Callable], callback: Callable):
        self.outputs = outputs
        self.callback = callback

        SequenceNumSync.__init__(self, len(outputs))

        self.packets = dict()

    def new_packet(self, packet: FramePacket, _=None):
        # print('new packet', packet, packet.name, 'seq num',packet.imgFrame.getSequenceNum())
        synced = self.sync(
            packet.imgFrame.getSequenceNum(),
            packet.name,
            packet
        )
        if synced:
            self.callback(synced)

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, _) -> List[XoutBase]:
        xouts = []
        for output in self.outputs:
            xoutbase: XoutBase = output(pipeline, device)
            xoutbase.setup_base(self.new_packet)
            xouts.append(xoutbase)

            xoutbase.setup_visualize(Visualizer(), xoutbase.name)

        return xouts


class TriggerActionConfig(BaseConfig):  # in future can extend SequenceNumSync
    trigger: Trigger
    action: Action

    def __init__(self, trigger: Trigger, action: Action):
        self.trigger = trigger
        self.action = action

    def setup(self, pipeline: dai.Pipeline, device, _) -> List[XoutBase]:
        if isinstance(self.action, RecordAction):
            controller = RecordController(self.trigger, self.action)
        else:  # TODO
            controller = TriggerActionController(self.trigger)

        trigger_xout: XoutBase = self.trigger.input(pipeline, device)
        action_xout: XoutBase = self.action.input(pipeline, device)

        # seems like necessary thing to do, if I want to work with packets (not msgs) from this xout,
        # and if I want to check for trigger inside actuator's thread, not inside check_queue
        trigger_xout.setup_base(controller.new_packet_trigger)
        action_xout.setup_base(controller.new_packet_action)

        # without setting visualizer up, XoutNnResults.on_callback() won't work
        trigger_xout.setup_visualize(visualizer=Visualizer(), name=trigger_xout.name, visualizer_enabled=False)

        controller.start(device, action_xout)  # device is not used

        # should be returned, if I want to work with its packets after
        # (not msgs, like in Record class, that's why RecordConfig returns just self.rec)
        return [trigger_xout, action_xout]


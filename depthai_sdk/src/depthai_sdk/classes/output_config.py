import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Callable, List, Union

import depthai as dai
from depthai_sdk.oak_outputs.syncing import SequenceNumSync
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase
from depthai_sdk.oak_outputs.xout.xout_depth import XoutDepth
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.record import Record
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.trigger_action import TriggerAction
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger
from depthai_sdk.visualize.visualizer import Visualizer

def find_new_name(name: str, names: List[str]):
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

class BaseConfig:
    @abstractmethod
    def setup(self, pipeline: dai.Pipeline, device: dai.Device, names: List[str]) -> List[XoutBase]:
        raise NotImplementedError()


class OutputConfig(BaseConfig):
    """
    Saves callbacks/visualizers until the device is fully initialized. I'll admit it's not the cleanest solution.
    """

    def __init__(self, output: Callable,
                 callback: Callable,
                 visualizer: Visualizer = None,
                 visualizer_enabled: bool = False,
                 record_path: Optional[str] = None):
        self.output = output  # Output of the component (a callback)
        self.callback = callback  # Callback that gets called after syncing
        self.visualizer = visualizer
        self.visualizer_enabled = visualizer_enabled
        self.record_path = record_path

    def setup(self, pipeline: dai.Pipeline, device, names: List[str]) -> List[XoutBase]:
        xoutbase: XoutBase = self.output(pipeline, device)
        xoutbase.setup_base(self.callback)

        if xoutbase.name in names:  # Stream name already exist, append a number to it
            xoutbase.name = find_new_name(xoutbase.name, names)
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


class RosStreamConfig(BaseConfig):
    outputs: List[Callable]
    ros = None

    def __init__(self, outputs: List[Callable]):
        self.outputs = outputs

    def setup(self, pipeline: dai.Pipeline, device, names: List[str]) -> List[XoutBase]:
        xouts: List[XoutFrames] = []
        for output in self.outputs:
            xoutbase: XoutFrames = output(pipeline, device)
            xoutbase.setup_base(None)
            xouts.append(xoutbase)

        envs = os.environ
        if 'ROS_VERSION' not in envs:
            raise Exception('ROS installation not found! Please install or source the ROS you would like to use.')

        version = envs['ROS_VERSION']
        if version == '1':
            raise Exception('ROS1 publsihing is not yet supported!')
            from depthai_sdk.integrations.ros.ros1_streaming import Ros1Streaming
            self.ros = Ros1Streaming()
        elif version == '2':
            from depthai_sdk.integrations.ros.ros2_streaming import Ros2Streaming
            self.ros = Ros2Streaming()
        else:
            raise Exception(f"ROS version '{version}' not recognized! Should be either '1' or '2'")

        self.ros.update(device, xouts)
        return [self]

    def new_msg(self, name, msg):
        self.ros.new_msg(name, msg)

    def check_queue(self, block):
        pass  # No queues

    def start_fps(self):
        pass

    # def is_ros1(self) -> bool:
    #     try:
    #         import rospy
    #         return True
    #     except:
    #         return False
    #
    # def is_ros2(self):
    #     try:
    #         import rclpy
    #         return True
    #     except:
    #         return False


class SyncConfig(BaseConfig, SequenceNumSync):
    def __init__(self, outputs: List[Callable], callback: Callable):
        self.outputs = outputs
        self.callback = callback

        SequenceNumSync.__init__(self, len(outputs))

        self.packets = dict()

    def new_packet(self, packet):
        # print('new packet', packet, packet.name, 'seq num',packet.imgFrame.getSequenceNum())
        synced = self.sync(
            packet.msg.getSequenceNum(),
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

        return xouts


class TriggerActionConfig(BaseConfig):
    def __init__(self, trigger: Trigger, action: Union[Callable, Action]):
        self.trigger = trigger
        self.action = Action(None, action) if isinstance(action, Callable) else action

    def setup(self, pipeline: dai.Pipeline, device, _) -> List[XoutBase]:
        controller = TriggerAction(self.trigger, self.action)

        trigger_xout: XoutBase = self.trigger.input(pipeline, device)
        trigger_xout.setup_base(controller.new_packet_trigger)
        # without setting visualizer up, XoutNnResults.on_callback() won't work
        trigger_xout.setup_visualize(visualizer=Visualizer(), name=trigger_xout.name, visualizer_enabled=False)

        if isinstance(self.action, Callable):
            return [trigger_xout]

        action_xouts = []
        if self.action.inputs:
            for output in self.action.inputs:
                xout: XoutBase = output(pipeline, device)
                xout.setup_base(controller.new_packet_action)
                action_xouts.append(xout)

        if isinstance(self.action, RecordAction):
            self.action.setup(device, action_xouts)  # creates writers for VideoRecorder()

        return [trigger_xout] + action_xouts

import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict

import depthai as dai
from depthai_sdk.oak_outputs.syncing import SequenceNumSync
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, ReplayStream
from depthai_sdk.oak_outputs.xout.xout_depth import XoutDepth
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.components.component import Component
from depthai_sdk.record import Record
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.trigger_action import TriggerAction
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger
from depthai_sdk.visualize.visualizer import Visualizer
from queue import Queue
from depthai_sdk.oak_outputs.fps import FPS

class BasePacketHandler:
    def __init__(self):
        self.fps = FPS()

    @abstractmethod
    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        raise NotImplementedError()

    def get_fps(self) -> float:
        return self.fps.fps()

    def _new_packet_callback(self, packet):
        """
        Callback from XoutBase. Don't override it. Does FPS counting and calls new_packet().
        """
        self.fps.next_iter()
        self.new_packet(packet)

    @abstractmethod
    def new_packet(self, packet):
        raise NotImplementedError()

    def close(self):
        """
        Used as a cleanup method (eg. close recording), other classes can override it.
        """
        pass

    def _get_outputs(self, output: Union[List, Callable, Component]) -> List[Callable]:
        if not isinstance(output, List):
            output = [output]

        for i in range(len(output)):
            if isinstance(output[i], Component):
                # Select default (main) output of the component
                output[i] = output[i].out.main
        return output

    def _create_xout(self, pipeline: dai.Pipeline, xout: XoutBase, xout_streams: Dict):
        # Assign which callback to call when packet is prepared
        xout.new_packet_callback = self._new_packet_callback

        for xstream in xout.xstreams():
            if xstream.name not in xout_streams:
                xout_streams[xstream.name] = []
                if not isinstance(xstream, ReplayStream):
                    xlink = pipeline.createXLinkOut()
                    xlink.setStreamName(xstream.name)
                    xstream.stream.link(xlink.input)
            xout_streams[xstream.name].append(xout.device_msg_callback)


# class OutputConfig(BaseConfig):
#     """
#     Saves callbacks/visualizers until the device is fully initialized. I'll admit it's not the cleanest solution.
#     """

#     def __init__(self, output: Callable,
#                  callback: Callable,
#                  visualizer: Visualizer = None,
#                  visualizer_enabled: bool = False,
#                  record_path: Optional[str] = None,
#                  main_thread: bool = True,
#                  queue: Queue = None
#                  ):
#         self.output = output  # Output of the component (a callback)
#         self.callback = callback  # Callback that gets called after syncing
#         self.visualizer = visualizer
#         self.visualizer_enabled = visualizer_enabled
#         self.record_path = record_path
#         self.main_thread = main_thread
#         self.queue = queue

#     def find_new_name(self, name: str, names: List[str]):
#         while True:
#             arr = name.split(' ')
#             num = arr[-1]
#             if num.isnumeric():
#                 arr[-1] = str(int(num) + 1)
#                 name = " ".join(arr)
#             else:
#                 name = f"{name} 2"
#             if name not in names:
#                 return name

#     def setup(self, pipeline: dai.Pipeline, device, names: List[str]) -> List[XoutBase]:
#         xoutbase: XoutBase = self.output(pipeline, device)


#         if xoutbase.name in names:  # Stream name already exist, append a number to it
#             xoutbase.name = self.find_new_name(xoutbase.name, names)
#         names.append(xoutbase.name)

#         recorder = None
#         if self.record_path:
#             recorder = VideoRecorder()

#             if isinstance(xoutbase, XoutDepth):
#                 raise NotImplementedError('Depth recording is not implemented yet.'
#                                           'Please use OakCamera.record() instead.')

#             recorder.update(Path(self.record_path), device, [xoutbase])

#         if self.visualizer:
#             xoutbase.setup_visualize(visualizer=self.visualizer,
#                                      visualizer_enabled=self.visualizer_enabled,
#                                      name=xoutbase.name)

#         if self.record_path:
#             xoutbase.setup_recorder(recorder=recorder)

#         # if self.queue is None:
#             # If queue is passed, oak.create_queue() was called. So we don't want
#             # to read frames from queue on oak.poll()
#             # return []
#         return [xoutbase]


class RecordPacketHandler(BasePacketHandler):
    def __init__(self, outputs, rec: Record):
        self.outputs = self._get_outputs(outputs)
        self.rec = rec

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        xouts: List[XoutBase] = []
        for output in self.outputs:
            xout = output(pipeline, device)
            xouts.append(xout)
            self._create_xout(pipeline, xout, xout_streams)

        self.rec.start(device, xouts)

    def close(self):
        self.rec.close()
        for xout in self.outputs:
            xout.close()

class CallbackPacketHandler(BasePacketHandler):
    def __init__(self, outputs, callback: Callable):
        self.outputs = self._get_outputs(outputs)
        self.callback = callback

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        for output in self.outputs:
            xout = output(pipeline, device)
            self._create_xout(pipeline, xout, xout_streams)

    def new_packet(self, packet):
        self.callback(packet)

class QueuePacketHandler(BasePacketHandler):
    def __init__(self, outputs, queue: Queue):
        self.outputs = self._get_outputs(outputs)
        self.queue = queue

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        for output in self.outputs:
            xout = output(pipeline, device)
            self._create_xout(pipeline, xout, xout_streams)

    def new_packet(self, packet):
        self.queue.put(packet)


class RosPacketHandler(BasePacketHandler):
    outputs: List[Callable]
    ros = None

    def __init__(self, outputs: List[Callable]):
        self.outputs = outputs

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        xouts = []
        for output in self.outputs:
            xouts.append(output(pipeline, device))

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

    def new_packet(self, packet):
        # self.ros.new_msg(name, msg)
        # TODO: implement
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

class TriggerActionPacketHandler(BasePacketHandler):
    def __init__(self, trigger: Trigger, action: Union[Callable, Action]):
        self.trigger = trigger
        self.action = Action(None, action) if isinstance(action, Callable) else action

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        controller = TriggerAction(self.trigger, self.action)

        trigger_xout: XoutBase = self.trigger.input(pipeline, device)
        # trigger_xout.setup_base(controller.new_packet_trigger)

        if isinstance(self.action, Callable):
            return [trigger_xout]

        action_xouts = []
        if self.action.inputs:
            for output in self.action.inputs:
                xout: XoutBase = output(pipeline, device)
                # xout.setup_base(controller.new_packet_action)
                action_xouts.append(xout)

        if isinstance(self.action, RecordAction):
            self.action.setup(device, action_xouts)  # creates writers for VideoRecorder()

        return [trigger_xout] + action_xouts

import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict

import depthai as dai
from depthai_sdk.classes.packets import DetectionPacket, TrackerPacket
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
import numpy as np

class BasePacketHandler:
    def __init__(self, main_thread=False):
        self.fps = FPS()
        self.queue = Queue(30) if main_thread else None

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
        if self.queue:
            self.queue.put(packet)
        else:
            self.new_packet(packet)

    def _poll(self):
        """
        Called from main thread.
        """
        if self.queue:
            while not self.queue.empty():
                packet = self.queue.get()
                self.new_packet(packet)

    @abstractmethod
    def new_packet(self, packet):
        raise NotImplementedError()

    def close(self):
        """
        Used as a cleanup method (eg. close recording), other classes can override it.
        """
        pass

    def _save_outputs(self, output: Union[List, Callable, Component]):
        if not isinstance(output, List):
            output = [output]

        for i in range(len(output)):
            if isinstance(output[i], Component):
                # Select default (main) output of the component
                output[i] = output[i].out.main

        self.outputs: List[Callable] = output

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


class VisualizePacketHandler(BasePacketHandler):
    def __init__(self,
                 outputs,
                 visualizer: Visualizer,
                 callback: Callable = None,
                 record_path: Optional[str] = None,
                 main_thread: bool = True,
                 ):
        self._save_outputs(outputs)

        if 1 < len(self.outputs) and record_path is not None:
            raise Exception('Recording multiple streams is not supported! Call oak.visualize(out, record_path="vid.mp4") for each stream separately')

        self.callback = callback  # Callback that gets called after syncing
        self.visualizer = visualizer
        self.record_path = record_path
        self.recorder = None
        # Main thread: if opencv visualizer, then we need to poll it
        super().__init__(main_thread)

    def new_packet(self, packet):
        self.visualizer.visualize(packet)

        if self.callback:
            self.callback(packet)

        if self.recorder:
            self.recorder.write(packet)


class RecordPacketHandler(BasePacketHandler):
    def __init__(self, outputs, rec: Record):
        self._save_outputs(outputs)
        self.rec = rec
        super().__init__()

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
    def __init__(self, outputs, callback: Callable, main_thread=False):
        self._save_outputs(outputs)
        self.callback = callback
        super().__init__(main_thread)

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        for output in self.outputs:
            xout = output(pipeline, device)
            self._create_xout(pipeline, xout, xout_streams)

    def new_packet(self, packet):
        self.callback(packet)

class QueuePacketHandler(BasePacketHandler):
    def __init__(self, outputs, queue: Queue):
        super().__init__()
        self._save_outputs(outputs)
        self.queue = queue

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        for output in self.outputs:
            xout = output(pipeline, device)
            self._create_xout(pipeline, xout, xout_streams)

    def new_packet(self, packet):
        # It won't be called, we just added this function to satisfy the abstract class
        pass


class RosPacketHandler(BasePacketHandler):
    def __init__(self, outputs):
        self._save_outputs(outputs)

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

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        xouts = []
        for output in self.outputs:
            xout = output(pipeline, device)
            self._create_xout(pipeline, xout, xout_streams)
            xouts.append(xout)

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

class StreamPacketHandler(BasePacketHandler):
    """
    TODO. API:
    oak.stream_rtsp([color, left, right], port=8888)
    oak.stream_webrtc(color, port=8881)

    Creates a server and just sends forward
    the frames. Doesn't use any queues.

    """
    pass

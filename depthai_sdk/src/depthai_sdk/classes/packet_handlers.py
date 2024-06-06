
import os
from abc import abstractmethod
from queue import Queue, Empty
from typing import Optional, Callable, List, Union, Dict

import depthai as dai

from depthai_sdk.classes.packets import BasePacket
from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.logger import LOGGER
from depthai_sdk.oak_outputs.fps import FPS
from depthai_sdk.oak_outputs.syncing import TimestampSync
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, ReplayStream
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.record import Record
from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.trigger_action import TriggerAction
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger
from depthai_sdk.visualize.visualizer import Visualizer


class BasePacketHandler:
    def __init__(self, main_thread=False):
        self.fps = FPS()
        self.queue = Queue(2) if main_thread else None
        self.outputs: List[ComponentOutput]
        self.sync = None

        self._packet_names = {}  # Check for duplicate packet name, raise error if found (user error)

    @abstractmethod
    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        raise NotImplementedError()

    def get_fps(self) -> float:
        return self.fps.fps()

    def _new_packet_callback(self, packet: BasePacket):
        """
        Callback from XoutBase. Don't override it. Does FPS counting and calls new_packet().
        """
        if self.sync is not None:
            packet = self.sync.sync(packet.get_timestamp(), packet.name, packet)
            if packet is None:
                return

        self.fps.next_iter()
        if self.queue:
            if self.queue.full():
                self.queue.get()  # Remove oldest packet
            self.queue.put(packet)
        else:
            self.new_packet(packet)

    def configure_syncing(self,
                          enable_sync: bool = True,
                          threshold_ms: int = 17):
        """
        If multiple outputs are used, then PacketHandler can do timestamp syncing of multiple packets
        before calling new_packet().
        Args:
            enable_sync: If True, then syncing is enabled.
            threshold_ms: Maximum time difference between packets in milliseconds.
        """
        if enable_sync:
            if len(self.outputs) < 2:
                LOGGER.error('Syncing requires at least 2 outputs! Skipping syncing.')
                return
            self.sync = TimestampSync(len(self.outputs), threshold_ms)

    def _poll(self):
        """
        Called from main thread.
        """
        if self.queue:
            try:
                packet = self.queue.get_nowait()
                self.new_packet(packet)
            except Empty:
                pass

    @abstractmethod
    def new_packet(self, packet):
        raise NotImplementedError()

    def close(self):
        """
        Used as a cleanup method (eg. close recording), other classes can override it.
        """
        pass

    def _save_outputs(self, output: Union[List, ComponentOutput, Component]):
        if not isinstance(output, List):
            output = [output]

        for i in range(len(output)):
            if isinstance(output[i], Component):
                # Select default (main) output of the component
                output[i] = output[i].out.main

        self.outputs = output

    def _create_xout(self,
                     pipeline: dai.Pipeline,
                     xout: XoutBase,
                     xout_streams: Dict,
                     custom_callback: Callable = None,
                     custom_packet_postfix: str = None):
        # Check for duplicate packet name, raise error if found (user error)
        if custom_packet_postfix:
            xout.set_packet_name_postfix(custom_packet_postfix)

        name = xout.get_packet_name()
        if name in self._packet_names:
            raise ValueError(
                f'User specified duplicate packet name "{name}"! Please specify unique names (or leave empty) for each component output.')
        self._packet_names[name] = True

        # Assign which callback to call when packet is prepared
        xout.new_packet_callback = custom_callback or self._new_packet_callback

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
            raise Exception('Recording multiple streams is not supported! '
                            'Call oak.visualize(out, record_path="vid.mp4") for each stream separately')

        self.callback = callback  # Callback that gets called after syncing
        self.visualizer = visualizer
        self.record_path = record_path
        self.recorder = None
        # Main thread: if opencv visualizer, then we need to poll it
        super().__init__(main_thread)

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        for output in self.outputs:
            xout: XoutBase = output(device)
            self._create_xout(pipeline, xout, xout_streams)

    def new_packet(self, packet: BasePacket):
        # Create visualizer objects for the visualizer. These objects will then be visualized
        # by the selected visualizer
        packet.prepare_visualizer_objects(self.visualizer)

        if self.callback:
            # Add self.visualizer to packet attributes
            packet.visualizer = self.visualizer
            self.callback(packet)
        else:
            self.visualizer.show(packet)

        if self.recorder:
            self.recorder.write(packet)

    def close(self):
        self.visualizer.close()


class RecordPacketHandler(BasePacketHandler):
    def __init__(self, outputs, recorder: Record):
        self._save_outputs(outputs)
        self.recorder = recorder
        super().__init__()

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        xouts: List[XoutFrames] = []
        for output in self.outputs:
            xout = output(device)
            xouts.append(xout)
            self._create_xout(pipeline, xout, xout_streams)

        self.recorder.start(device, xouts)

    def new_packet(self, packet: BasePacket):
        self.recorder.write(packet)

    def close(self):
        self.recorder.close()


class CallbackPacketHandler(BasePacketHandler):
    def __init__(self, outputs, callback: Callable, main_thread=False):
        self._save_outputs(outputs)
        self.callback = callback
        super().__init__(main_thread)

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        for output in self.outputs:
            xout = output(device)
            self._create_xout(pipeline, xout, xout_streams)

    def new_packet(self, packet):
        self.callback(packet)


class QueuePacketHandler(BasePacketHandler):
    def __init__(self, outputs, max_size: int):
        super().__init__()
        self._save_outputs(outputs)
        self.queue = Queue(max_size)

    def get_queue(self) -> Queue:
        return self.queue

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        for output in self.outputs:
            xout = output(device)
            self._create_xout(pipeline, xout, xout_streams)

    def configure_syncing(self,
                          enable_sync: bool = True,
                          threshold_ms: int = 17) -> 'QueuePacketHandler':
        """
        If multiple outputs are used, then PacketHandler can do timestamp syncing of multiple packets
        before calling new_packet().
        Args:
            enable_sync: If True, then syncing is enabled.
            threshold_ms: Maximum time difference between packets in milliseconds.
        """
        super().configure_syncing(enable_sync, threshold_ms)
        return self

    def new_packet(self, packet):
        # It won't be called, we just added this function to satisfy the abstract class
        pass


class RosPacketHandler(BasePacketHandler):
    def __init__(self, outputs):
        super().__init__()
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
            xout = output(device)
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
        super().__init__()
        self.trigger = trigger
        self.action = Action(None, action) if isinstance(action, Callable) else action
        self.controller = TriggerAction(self.trigger, self.action)

    def setup(self, pipeline: dai.Pipeline, device: dai.Device, xout_streams: Dict[str, List]):
        trigger_xout: XoutBase = self.trigger.input(device)
        self._create_xout(pipeline=pipeline,
                          xout=trigger_xout,
                          xout_streams=xout_streams,
                          custom_callback=self.controller.new_packet_trigger,
                          custom_packet_postfix='trigger')

        if isinstance(self.action, Callable):
            self._save_outputs([trigger_xout])
            return

        action_xouts = []
        if self.action.inputs:
            for output in self.action.inputs:
                xout: XoutBase = output(device)
                xout.new_packet_callback = self.controller.new_packet_action
                self._create_xout(pipeline=pipeline,
                                  xout=xout,
                                  xout_streams=xout_streams,
                                  custom_callback=self.controller.new_packet_action,
                                  custom_packet_postfix='action')
                action_xouts.append(xout)

        if isinstance(self.action, RecordAction):
            self.action.setup(device, action_xouts)  # creates writers for VideoRecorder()

        self._save_outputs([trigger_xout] + action_xouts)

    def new_packet(self, packet):
        pass


class StreamPacketHandler(BasePacketHandler):
    """
    TODO. API:
    oak.stream_rtsp([color, left, right], port=8888)
    oak.stream_webrtc(color, port=8881)

    Creates a server and just sends forward
    the frames. Doesn't use any queues.

    """
    pass

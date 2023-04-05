import logging
import warnings
from typing import List, Optional, Union

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import DepthPacket, PointcloudPacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.components.pointcloud_helper import create_xyz

try:
    import cv2
except ImportError:
    cv2 = None


class XoutPointcloud(XoutFrames):
    def __init__(self,
                 device: dai.Device,
                 depth_frames: StreamXout,
                 fps: int,
                 color_frames: Optional[StreamXout] = None):

        self.color_frames = color_frames
        XoutFrames.__init__(self, frames=depth_frames, fps=fps)
        self.name = 'Pointcloud'
        self.fps = fps
        self.device = device
        self.xyz = None

        self.msgs = dict()

    def visualize(self, packet: DepthPacket):
        pass

    def xstreams(self) -> List[StreamXout]:
        if self.color_frames is not None:
            return [self.frames, self.color_frames]
        return [self.frames]

    def new_msg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams:
            return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()

        if name == self.frames.name:
            self.msgs[seq][name] = msg
        elif name == self.color_frames.name:
            self.msgs[seq][name] = msg
        else:
            raise ValueError('Message from unknown stream name received by XOutPointcloud!')

        if len(self.msgs[seq]) == len(self.xstreams()):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            depth_frame: dai.ImgFrame = self.msgs[seq][self.frames.name]

            color_frame = None
            if self.color_frames is not None:
                color_frame: dai.ImgFrame = self.msgs[seq][self.color_frames.name]

            if self.xyz is None:
                self.xyz = create_xyz(self.device, depth_frame.getWidth(), depth_frame.getHeight())

            pcl = self.xyz * np.expand_dims(np.array(depth_frame.getFrame()), axis = -1)

            # TODO: postprocessing

            packet = PointcloudPacket(
                self.get_packet_name(),
                pcl,
                depth_map=depth_frame,
                color_frame=color_frame,
                visualizer=self._visualizer
            )
            self.queue.put(packet, block=False)

            new_msgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    new_msgs[name] = msg
            self.msgs = new_msgs

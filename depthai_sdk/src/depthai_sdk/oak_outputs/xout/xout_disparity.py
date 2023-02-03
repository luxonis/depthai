from typing import List

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import DepthPacket
from depthai_sdk.oak_outputs.xout import Clickable
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.visualize.configs import StereoColor

try:
    import cv2
except ImportError:
    cv2 = None


class XoutDisparity(XoutFrames, Clickable):
    name: str = "Disparity"

    def __init__(self,
                 disparity_frames: StreamXout,
                 mono_frames: StreamXout,
                 max_disp: float,
                 fps: float,
                 colorize: StereoColor = None,
                 colormap: int = None,
                 use_wls_filter: bool = None,
                 wls_level: 'WLSLevel' = None,
                 wls_lambda: float = None,
                 wls_sigma: float = None):
        self.mono_frames = mono_frames

        self.multiplier = 255.0 / max_disp
        self.fps = fps

        self.colorize = colorize
        self.colormap = colormap

        self.use_wls_filter = use_wls_filter

        # Prefer to use WLS level if set, otherwise use lambda and sigma
        if wls_level and use_wls_filter:
            print(f'Using WLS level: {wls_level.name} (lambda: {wls_level.value[0]}, sigma: {wls_level.value[1]})')
            self.wls_lambda = wls_level.value[0]
            self.wls_sigma = wls_level.value[1]
        else:
            self.wls_lambda = wls_lambda
            self.wls_sigma = wls_sigma

        if self.use_wls_filter:
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

        self.msgs = dict()

        XoutFrames.__init__(self, frames=disparity_frames, fps=fps)
        Clickable.__init__(self, decay_step=int(self.fps))

    def visualize(self, packet: DepthPacket):
        frame = packet.frame
        disparity_frame = (frame * self.multiplier).astype(np.uint8)

        stereo_config = self._visualizer.config.stereo

        if self.use_wls_filter or stereo_config.wls_filter:
            self.wls_filter.setLambda(self.wls_lambda or stereo_config.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma or stereo_config.wls_sigma)
            disparity_frame = self.wls_filter.filter(disparity_frame, packet.mono_frame.getCvFrame())

        colorize = self.colorize or stereo_config.colorize
        colormap = self.colormap or stereo_config.colormap
        if colorize == StereoColor.GRAY:
            packet.frame = disparity_frame
        elif colorize == StereoColor.RGB:
            packet.frame = cv2.applyColorMap(disparity_frame, colormap)
        elif colorize == StereoColor.RGBD:
            packet.frame = cv2.applyColorMap(
                (disparity_frame * 0.5 + packet.mono_frame.getCvFrame() * 0.5).astype(np.uint8), colormap
            )

        if self._visualizer.config.output.clickable:
            cv2.namedWindow(self.name)
            cv2.setMouseCallback(self.name, self.on_click_callback, param=[disparity_frame])

            if self.buffer:
                x, y = self.buffer[2]
                self._visualizer.add_circle(coords=(x, y), radius=3, color=(255, 255, 255), thickness=-1)
                self._visualizer.add_text(
                    text=f'{self.buffer[1]}',
                    coords=(x, y - 10)
                )

        super().visualize(packet)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.mono_frames]

    def new_msg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams:
            return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()

        if name == self.frames.name:
            self.msgs[seq][name] = msg
        elif name == self.mono_frames.name:
            self.msgs[seq][name] = msg
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if len(self.msgs[seq]) == len(self.xstreams()):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            packet = DepthPacket(
                self.get_packet_name(),
                self.msgs[seq][self.frames.name],
                self.msgs[seq][self.mono_frames.name],
                self._visualizer
            )
            self.queue.put(packet, block=False)

            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs

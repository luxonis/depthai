from typing import List

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import DepthPacket
from depthai_sdk.oak_outputs.xout import Clickable
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.visualize.configs import StereoColor
from depthai_sdk.visualize.visualizer import Visualizer

try:
    import cv2
except ImportError:
    cv2 = None


class XoutDepth(XoutFrames, Clickable):
    def __init__(self,
                 device: dai.Device,
                 frames: StreamXout,
                 fps: float,
                 mono_frames: StreamXout,
                 colorize: StereoColor = None,
                 colormap: int = None,
                 use_wls_filter: bool = None,
                 wls_level: 'WLSLevel' = None,
                 wls_lambda: float = None,
                 wls_sigma: float = None):
        self.mono_frames = mono_frames
        XoutFrames.__init__(self, frames=frames, fps=fps)
        Clickable.__init__(self, decay_step=int(self.fps))

        self.name = 'Depth'
        self.fps = fps
        self.device = device

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

        self.wls_filter = None
        self.msgs = dict()

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        super().setup_visualize(visualizer, visualizer_enabled, name)

        if self._visualizer.config.stereo.wls_filter or self.use_wls_filter:
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

    def visualize(self, packet: DepthPacket):
        depth_frame = packet.imgFrame.getFrame()

        stereo_config = self._visualizer.config.stereo

        if self.use_wls_filter or stereo_config.wls_filter:
            self.wls_filter.setLambda(stereo_config.wls_lambda)
            self.wls_filter.setSigmaColor(stereo_config.wls_sigma)
            depth_frame = self.wls_filter.filter(depth_frame, packet.mono_frame.getCvFrame())

        depth_frame_color = cv2.normalize(depth_frame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
        depth_frame_color = cv2.equalizeHist(depth_frame_color)

        colorize = self.colorize or stereo_config.colorize
        colormap = self.colormap or stereo_config.colormap
        if colorize == StereoColor.GRAY:
            packet.frame = depth_frame_color
        elif colorize == StereoColor.RGB:
            packet.frame = cv2.applyColorMap(depth_frame_color, colormap)
        elif colorize == StereoColor.RGBD:
            packet.frame = cv2.applyColorMap(
                (depth_frame_color * 0.5 + packet.mono_frame.getCvFrame() * 0.5).astype(np.uint8),
                stereo_config.colormap
            )

        if self._visualizer.config.output.clickable:
            cv2.namedWindow(self.name)
            cv2.setMouseCallback(self.name, self.on_click_callback, param=[depth_frame])

            if self.buffer:
                x, y = self.buffer[2]
                self._visualizer.add_circle(coords=(x, y), radius=3, color=(255, 255, 255), thickness=-1)
                self._visualizer.add_text(
                    text=f'{self.buffer[1] / 10} cm',
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

            new_msgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    new_msgs[name] = msg
            self.msgs = new_msgs

from typing import Optional

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import DepthPacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_disparity import XoutDisparity
from depthai_sdk.visualize.configs import StereoColor

try:
    import cv2
except ImportError:
    cv2 = None


class XoutDepth(XoutDisparity):
    def __init__(self,
                 device: dai.Device,
                 frames: StreamXout,
                 dispScaleFactor: float,
                 fps: float,
                 mono_frames: Optional[StreamXout],
                 colorize: StereoColor = None,
                 colormap: int = None,
                 wls_config: dict = None,
                 ir_settings: dict = None):
        self.name = 'Depth'
        super().__init__(device=device,
                         frames=frames,
                         disp_factor=255 / 95,
                         fps=fps,
                         mono_frames=mono_frames,
                         colorize=colorize,
                         colormap=colormap,
                         wls_config=wls_config,
                         ir_settings=ir_settings)

        self.disp_scale_factor = dispScaleFactor

    def visualize(self, packet: DepthPacket):
        # Convert depth to disparity for nicer visualization
        packet.depth_map = packet.frame.copy()
        with np.errstate(divide='ignore'):
            disp = self.disp_scale_factor / packet.frame

        disp[disp == np.inf] = 0

        packet.frame = np.round(disp).astype(np.uint8)
        super().visualize(packet)

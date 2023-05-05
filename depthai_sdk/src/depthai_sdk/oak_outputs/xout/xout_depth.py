import logging
import warnings
from typing import Optional, List

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import DepthPacket
from depthai_sdk.oak_outputs.xout import Clickable
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.visualize.configs import StereoColor
from depthai_sdk.visualize.visualizer import Visualizer
from depthai_sdk.oak_outputs.xout.xout_disparity import XoutDisparity

try:
    import cv2
except ImportError:
    cv2 = None


class XoutDepth(XoutDisparity):
    def __init__(self,
                 device: dai.Device,
                 frames: StreamXout,
                 stereo: dai.node.StereoDepth,
                 fps: float,
                 mono_frames: Optional[StreamXout],
                 colorize: StereoColor = None,
                 colormap: int = None,
                 use_wls_filter: bool = None,
                 wls_level: 'WLSLevel' = None,
                 wls_lambda: float = None,
                 wls_sigma: float = None):

        self.name = 'Depth'
        super().__init__(
                        frames=frames,
                        stereo=stereo,
                        fps=fps,
                        mono_frames=mono_frames,
                        colorize=colorize,
                        colormap=colormap,
                        use_wls_filter=use_wls_filter,
                        wls_level=wls_level,
                        wls_lambda=wls_lambda,
                        wls_sigma=wls_sigma)

        calib = device.readCalibration()
        baseline = calib.getBaselineDistance() * 10 # cm to mm

        align: dai.CameraBoardSocket = stereo.properties.depthAlignCamera
        if align == dai.CameraBoardSocket.AUTO:
            align = dai.CameraBoardSocket.RIGHT

        intrinsics = calib.getCameraIntrinsics(align)
        focal = intrinsics[0][0] # focal length in pixels
        self.dispScaleFactor = baseline * focal

    def visualize(self, packet: DepthPacket):
        # Convert depth to disparity for nicer visualization
        packet.depth_map = packet.frame.copy()
        with np.errstate(divide='ignore'):
            disp = self.dispScaleFactor / packet.frame

        disp[disp==np.inf] = 0
        packet.frame = np.round(disp).astype(np.uint8)
        super().visualize(packet)

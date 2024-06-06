from typing import Optional, Dict

import depthai as dai

from depthai_sdk.classes.packets import DisparityDepthPacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_disparity import XoutDisparity
from depthai_sdk.visualize.configs import StereoColor


class XoutDisparityDepth(XoutDisparity):
    def __init__(self,
                 device: dai.Device,
                 frames: StreamXout,
                 dispScaleFactor: float,
                 aligned_frame: Optional[StreamXout],
                 colorize: StereoColor = None,
                 colormap: int = None,
                 ir_settings: dict = None,
                 confidence_map: StreamXout = None):
        self.name = 'Depth'
        super().__init__(device=device,
                         frames=frames,
                         disp_factor=255 / 95,
                         aligned_frame=aligned_frame,
                         colorize=colorize,
                         colormap=colormap,
                         ir_settings=ir_settings,
                         confidence_map=confidence_map)

        self.disp_scale_factor = dispScaleFactor

    def package(self, msgs: Dict) -> DisparityDepthPacket:
        aligned_frame = msgs[self.aligned_frame.name] if self.aligned_frame else None
        confidence_map = msgs[self.confidence_map.name] if self.confidence_map else None
        return DisparityDepthPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            colorize=self.colorize,
            colormap=self.colormap,
            aligned_frame=aligned_frame,
            disp_scale_factor=self.disp_scale_factor,
            confidence_map=confidence_map
        )

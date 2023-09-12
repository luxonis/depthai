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
                 mono_frames: Optional[StreamXout],
                 colorize: StereoColor = None,
                 colormap: int = None,
                 ir_settings: dict = None,
                 confidence_map: StreamXout = None):
        self.name = 'Depth'
        super().__init__(device=device,
                         frames=frames,
                         disp_factor=255 / 95,
                         mono_frames=mono_frames,
                         colorize=colorize,
                         colormap=colormap,
                         ir_settings=ir_settings,
                         confidence_map=confidence_map)

        self.disp_scale_factor = dispScaleFactor

    def package(self, msgs: Dict) -> DisparityDepthPacket:
        mono_frame = msgs[self.mono_frames.name] if self.mono_frames else None
        return DisparityDepthPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            colorize=self.colorize,
            colormap=self.colormap,
            mono_frame=mono_frame,
            disp_scale_factor=self.disp_scale_factor,
        )

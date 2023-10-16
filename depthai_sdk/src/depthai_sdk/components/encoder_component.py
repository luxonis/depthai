import logging
from typing import Dict, List, Optional, Union

import depthai as dai
from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.components.parser import parse_encode


class EncoderComponent(Component):
    def __init__(
        self,
        device: dai.Device,
        pipeline: dai.Pipeline,
        input: Union[CameraComponent, StereoComponent],
        codec: Union[str, dai.VideoEncoderProperties.Profile],
    ) -> None:
        super().__init__()

        self._device = device
        self._pipeline = pipeline

        input.ensure_encoder_compatible_size()

        self.node = pipeline.create(dai.node.VideoEncoder)
        self.node.setDefaultProfilePreset(input.get_fps(), parse_encode(codec))

        _get_node_out(input).link(self.node.input)

    def config_encoder_h26x(
        self,
        rate_control_mode: Optional[dai.VideoEncoderProperties.RateControlMode] = None,
        keyframe_freq: Optional[int] = None,
        bitrate_kbps: Optional[int] = None,
        num_b_frames: Optional[int] = None,
    ):
        if self.node.getProfile() not in [
            dai.VideoEncoderProperties.Profile.H264_BASELINE,
            dai.VideoEncoderProperties.Profile.H264_HIGH,
            dai.VideoEncoderProperties.Profile.H264_MAIN,
            dai.VideoEncoderProperties.Profile.H265_MAIN,
        ]:
            raise ValueError(f"Encoder profile {self.node.getProfile()} is not H.26x")
        if rate_control_mode is not None:
            self.node.setRateControlMode(rate_control_mode)
        if keyframe_freq is not None:
            self.node.setKeyframeFrequency(keyframe_freq)
        if bitrate_kbps is not None:
            self.node.setBitrateKbps(bitrate_kbps)
        if num_b_frames is not None:
            self.node.setNumBFrames(num_b_frames)

    def config_encoder_mjpeg(
        self,
        quality: Optional[int] = None,
        lossless: bool = False,
    ):
        if self.node.getProfile() != dai.VideoEncoderProperties.Profile.MJPEG:
            raise ValueError(f"Encoder profile {self.node.getProfile()} is not MJPEG")
        if quality is not None:
            self.node.setQuality(quality)
        if lossless is not None:
            self.node.setLossless(lossless)


def _get_node_out(component: Union[CameraComponent, StereoComponent]) -> dai.Node.Output:
    if isinstance(component, CameraComponent):
        if isinstance(component.node, dai.node.ColorCamera):
            return component.node.video
        elif isinstance(component.node, dai.node.MonoCamera):
            return component.node.out
        raise ValueError(f"Unknown camera node: {component.node}")
    elif isinstance(component, StereoComponent):
        if component.colormap_manip:
            return component.colormap_manip.out
        return component.node.disparity
    raise ValueError(f"Unknown component: {component}")

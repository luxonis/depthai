from typing import List, Optional, Union

import depthai as dai

from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.parser import parse_camera_socket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout, XoutBase
from depthai_sdk.oak_outputs.xout.xout_depth import XoutDisparityDepth
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.components.tof_control import ToFControl


class ToFComponent(Component):

    _align: Optional[dai.node.ImageAlign] = None

    def __init__(
        self,
        device: dai.Device,
        pipeline: dai.Pipeline,
        source: Union[str, dai.CameraBoardSocket, None] = None,
        align_to: Optional[CameraComponent] = None,
    ):
        super().__init__()
        self.out = self.Out(self)
        self._pipeline = pipeline

        if source is None:
            source = self._find_tof(device)
        elif isinstance(source, str):
            source = parse_camera_socket(source)
        elif isinstance(source, dai.CameraBoardSocket):
            pass  # This is what we want
        else:
            raise ValueError("source must be either None, str, or CameraBoardSocket!")

        self.control = ToFControl(device)
        self.camera_node = pipeline.create(dai.node.ColorCamera)
        self.camera_node.setBoardSocket(source)
        self.camera_socket = source

        self.node = pipeline.create(dai.node.ToF)
        self._control_in = pipeline.create(dai.node.XLinkIn)
        self.camera_node.raw.link(self.node.input)
        self._control_in.setStreamName(f"{self.node.id}_inputControl")
        self._control_in.out.link(self.node.inputConfig)

        if align_to is not None:
            self._align = pipeline.create(dai.node.ImageAlign)
            self._align_to_output = align_to.node.isp
            self.node.depth.link(self._align.input)
            self._align_to_output.link(self._align.inputAlignTo)

    def _find_tof(self, device: dai.Device) -> dai.CameraBoardSocket:
        # Use the first ToF sensor, usually, there will only be one
        features = device.getConnectedCameraFeatures()
        for cam_sensor in features:
            if dai.CameraSensorType.TOF in cam_sensor.supportedTypes:
                return cam_sensor.socket
        raise RuntimeError(
            f"No ToF sensor found on this camera! {device.getConnectedCameraFeatures()}"
        )

    def set_align_to(self, align_to: CameraComponent) -> None:
        if self._align is None:
            self._align = self._pipeline.create(dai.node.ImageAlign)
            self.node.depth.link(self._align.input)
            if align_to.is_mono():
                align_to.node.out.link(self._align.inputAlignTo)
            else:
                align_to.node.isp.link(self._align.inputAlignTo)

    def on_pipeline_started(self, device: dai.Device) -> None:
        self.control.set_input_queue(
            device.getInputQueue(self._control_in.getStreamName())
        )

    class Out:

        class DepthOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                return XoutDisparityDepth(
                    device=device,
                    frames=StreamXout(
                        (
                            self._comp.node.depth
                            if self._comp._align is None
                            else self._comp._align.outputAligned
                        ),
                        "tof_depth",
                    ),
                    aligned_frame=StreamXout(self._comp._align_to_output, "aligned_stream") if self._comp._align else None,
                    dispScaleFactor=9500,
                    ir_settings={"auto_mode": False},
                ).set_comp_out(self)

        class AmplitudeOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                return XoutFrames(
                    frames=StreamXout(self._comp.node.amplitude, "tof_amplitude")
                ).set_comp_out(self)

        class ErrorOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                return XoutFrames(
                    frames=StreamXout(self._comp.node.error, "tof_error")
                ).set_comp_out(self)

        def __init__(self, tof_component: "ToFComponent"):
            self.depth = self.DepthOut(tof_component)
            self.amplitude = self.AmplitudeOut(tof_component)
            self.error = self.ErrorOut(tof_component)
            self.main = self.depth

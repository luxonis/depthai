from typing import List, Union

import depthai as dai

from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.components.parser import parse_camera_socket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout, XoutBase
from depthai_sdk.oak_outputs.xout.xout_depth import XoutDisparityDepth
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames


class ToFComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 source: Union[str, dai.CameraBoardSocket, None] = None,
                 ):
        super().__init__()
        self.out = self.Out(self)
        self._pipeline = pipeline

        if source is None:
            source = self._find_tof(device)
        elif isinstance(source, str):
            source = parse_camera_socket(source)
        elif isinstance(source, dai.CameraBoardSocket):
            pass # This is what we want
        else:
            raise ValueError('source must be either None, str, or CameraBoardSocket!')

        self.camera_node = pipeline.create(dai.node.ColorCamera)
        self.camera_node.setBoardSocket(source)
        self.camera_socket = source

        self.node = pipeline.create(dai.node.ToF)
        self.camera_node.raw.link(self.node.input)

    def _find_tof(self, device: dai.Device) -> dai.CameraBoardSocket:
        # Use the first ToF sensor, usually, there will only be one
        features = device.getConnectedCameraFeatures()
        for cam_sensor in features:
            if dai.CameraSensorType.TOF in cam_sensor.supportedTypes:
                return cam_sensor.socket
        raise RuntimeError(f'No ToF sensor found on this camera! {device.getConnectedCameraFeatures()}')

    class Out:

        class DepthOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                return XoutDisparityDepth(
                    device=device,
                    frames=StreamXout(self._comp.node.depth, "tof_depth"),
                    dispScaleFactor=9500,
                    mono_frames=None,
                    ir_settings={
                        "auto_mode": False
                    }
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

        def __init__(self, tof_component: 'ToFComponent'):
            self.depth = self.DepthOut(tof_component)
            self.amplitude = self.AmplitudeOut(tof_component)
            self.error = self.ErrorOut(tof_component)
            self.main = self.depth

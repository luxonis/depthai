from typing import List, Union

import depthai as dai

from depthai_sdk.components.component import Component, XoutBase
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_depth import XoutDepth
from depthai_sdk.components.parser import parse_camera_socket

class ToFComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 source: Union[str, dai.CameraBoardSocket, None] = None,
                 ):
        super().__init__()
        self.out = self.Out(self)

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

        tofConfig = self.node.initialConfig.get()
        # tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MIN
        tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MAX
        tofConfig.depthParams.avgPhaseShuffle = False
        tofConfig.depthParams.minimumAmplitude = 3.0
        self.node.initialConfig.set(tofConfig)

    def _find_tof(self, device: dai.Device) -> dai.CameraBoardSocket:
        # Use the first ToF sensor, usually, there will only be one
        features = device.getConnectedCameraFeatures()
        for cam_sensor in features:
            if dai.CameraSensorType.TOF in cam_sensor.supportedTypes:
                return cam_sensor.socket
        raise RuntimeError(f'No ToF sensor found on this camera! {device.getConnectedCameraFeatures()}')

    class Out:
        def __init__(self, component: 'ToFComponent'):
            self._comp = component

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Default output
            """
            return self.depth(pipeline, device)

        def depth(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            tof_out = XoutDepth(
                frames=StreamXout(self._comp.node.id, self._comp.node.depth),
                dispScaleFactor=9500,
                fps=30,
                mono_frames=None
                )
            return self._comp._create_xout(pipeline, tof_out)

        def amplitude(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            out = StreamXout(self._comp.node.id, self._comp.node.amplitude)
            tof_out = XoutFrames(out)
            return self._comp._create_xout(pipeline, tof_out)

        def error(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            out = StreamXout(self._comp.node.id, self._comp.node.error)
            tof_out = XoutFrames(out)
            return self._comp._create_xout(pipeline, tof_out)

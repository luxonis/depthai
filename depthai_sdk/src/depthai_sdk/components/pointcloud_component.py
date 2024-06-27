from typing import Optional, Union, Any

import depthai as dai

from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.components.tof_component import ToFComponent
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from depthai_sdk.oak_outputs.xout.xout_pointcloud import XoutPointcloud
from depthai_sdk.replay import Replay


class PointcloudComponent(Component):

    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 depth_input: Union[None, StereoComponent, ToFComponent, dai.node.StereoDepth, dai.Node.Output] = None,
                 colorize: Optional[CameraComponent] = None,
                 replay: Optional[Replay] = None,
                 args: Any = None):
        """
        Args:
            pipeline (dai.Pipeline): DepthAI pipeline
            replay (Replay object, optional): Replay
            args (Any, optional): Use user defined arguments when constructing the pipeline
        """
        super().__init__()
        self.out = self.Out(self)

        self.depth: dai.Node.Output  # depth output

        self.colorize_comp: Optional[CameraComponent] = colorize

        self._replay: Optional[Replay] = replay

        # Depth aspect
        if depth_input is None:
            depth_input = StereoComponent(device, pipeline, replay=replay, args=args)
            depth_input.config_stereo(lr_check=True, subpixel=True, subpixel_bits=3, confidence=230)
            depth_input.node.initialConfig.setNumInvalidateEdgePixels(20)

            config = depth_input.node.initialConfig.get()
            config.postProcessing.speckleFilter.enable = True
            config.postProcessing.speckleFilter.speckleRange = 50
            config.postProcessing.temporalFilter.enable = True
            config.postProcessing.spatialFilter.enable = True
            config.postProcessing.spatialFilter.holeFillingRadius = 2
            config.postProcessing.spatialFilter.numIterations = 1
            config.postProcessing.thresholdFilter.minRange = 400  # 40cm
            config.postProcessing.thresholdFilter.maxRange = 20000  # 20m
            config.postProcessing.decimationFilter.decimationFactor = 2
            config.postProcessing.decimationFilter.decimationMode = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
            depth_input.node.initialConfig.set(config)

            if self.colorize_comp is not None:
                # Align to colorize node
                depth_input.config_stereo(align=self.colorize_comp)

        if isinstance(depth_input, StereoComponent):
            depth_input = depth_input.node
        elif isinstance(depth_input, ToFComponent):
            if depth_input._align is not None:
                self.depth = depth_input._align.outputAligned
            else:
                self.depth = depth_input.node.depth

        if isinstance(depth_input, dai.node.StereoDepth):
            self.depth = depth_input.depth
        elif isinstance(depth_input, dai.Node.Output):
            self.depth = depth_input

    def config_postprocessing(self) -> None:
        """
        Configures postprocessing options.
        """
        raise NotImplementedError("config_postprocessing() not yet implemented")

    class Out:
        class PointcloudOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                colorize = None
                if self._comp.colorize_comp is not None:
                    colorize = StreamXout(self._comp.colorize_comp.stream, name="Color")
                return XoutPointcloud(device,
                                      StreamXout(self._comp.depth),
                                      color_frames=colorize).set_comp_out(self)

        def __init__(self, component: 'PointcloudComponent'):
            self.pointcloud = self.PointcloudOut(component)
            self.main = self.pointcloud

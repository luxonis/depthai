import logging
import warnings
from typing import Optional, Union, Any, Dict, Tuple

import cv2
import depthai as dai
import numpy as np
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from depthai_sdk.oak_outputs.xout.xout_pointcloud import XoutPointcloud
from depthai_sdk.replay import Replay


class PointcloudComponent(Component):

    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 stereo: Union[None, StereoComponent, dai.node.StereoDepth, dai.Node.Output] = None,
                 colorize: Optional[CameraComponent] = None,
                 replay: Optional[Replay] = None,
                 args: Any = None,
                 name: Optional[str] = None):
        """
        Args:
            pipeline (dai.Pipeline): DepthAI pipeline
            replay (Replay object, optional): Replay
            args (Any, optional): Use user defined arguments when constructing the pipeline
            name (str, optional): Name of the output stream
        """
        super().__init__()
        self.out = self.Out(self)

        self.stereo_depth_node: dai.node.StereoDepth
        self.depth: dai.Node.Output # Depth node output

        self.colorize_comp: Optional[CameraComponent] = colorize

        self.name = name

        self._replay: Optional[Replay] = replay

        # Colorization aspect
        if colorize is None:
            self.colorize_comp = CameraComponent(device, pipeline, source='color', replay=replay, args=args)

        if isinstance(self.colorize_comp, CameraComponent):
            self.colorize_comp.config_color_camera(isp_scale=(2,5))

        # Depth aspect
        if stereo is None:
            stereo = StereoComponent(device, pipeline, replay=replay, args=args)
            stereo.config_stereo(lr_check=True, subpixel=True, subpixel_bits=3, confidence=230)
            stereo.node.initialConfig.setNumInvalidateEdgePixels(20)

            config = stereo.node.initialConfig.get()
            config.postProcessing.speckleFilter.enable = True
            config.postProcessing.speckleFilter.speckleRange = 50
            config.postProcessing.temporalFilter.enable = True
            config.postProcessing.spatialFilter.enable = True
            config.postProcessing.spatialFilter.holeFillingRadius = 2
            config.postProcessing.spatialFilter.numIterations = 1
            config.postProcessing.thresholdFilter.minRange = 400 # 40cm
            config.postProcessing.thresholdFilter.maxRange = 20000 # 20m
            config.postProcessing.decimationFilter.decimationFactor = 2
            config.postProcessing.decimationFilter.decimationMode = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
            stereo.node.initialConfig.set(config)

            if self.colorize_comp is not None:
                # Align to colorize node
                stereo.config_stereo(align=self.colorize_comp)

        if isinstance(stereo, StereoComponent):
            stereo = stereo.node

        if isinstance(stereo, dai.node.StereoDepth):
            self.stereo_depth_node = stereo
            self.depth = stereo.depth
        elif isinstance(stereo, dai.Node.Output):
            self.stereo_depth_node = stereo.getParent()
            self.depth = stereo


    def config_postprocessing(self,
                              ) -> None:
        """
        Configures postprocessing options.

        Args:
        """
        pass

    class Out:
        def __init__(self, component: 'PointcloudComponent'):
            self._comp = component

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            return self.pointcloud(pipeline, device)

        def pointcloud(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            colorize = None
            if self._comp.colorize_comp is not None:
                colorize = StreamXout(self._comp.colorize_comp.node.id, self._comp.colorize_comp.stream, name="Color")

            out = XoutPointcloud(device,
                                 StreamXout(self._comp.stereo_depth_node.id, self._comp.depth, name=self._comp.name),
                                 color_frames=colorize,
                                 fps=30
                                 )
            return self._comp._create_xout(pipeline, out)

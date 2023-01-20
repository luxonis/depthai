import os
from pathlib import Path
from string import Template
from typing import Tuple, Optional, List

import depthai as dai

from depthai_sdk.types import GenericNeuralNetwork


class MultiStageConfig:
    def __init__(self,
                 debug: bool,
                 labels: Optional[List[int]] = None,
                 scale_bb: Optional[Tuple[int, int]] = None):
        self.debug = debug
        self.labels = labels
        self.scale_bb = scale_bb


class MultiStageNN:
    def __init__(self,
                 pipeline: dai.Pipeline,
                 detection_node: GenericNeuralNetwork,  # Object detection node
                 high_res_frames: dai.Node.Output,
                 size: Tuple[int, int],
                 debug=False
                 ) -> None:
        """
        Args:
            pipeline (dai.Pipeline): Pipeline object
            detection_node (GenericNeuralNetwork): Object detection NN
            high_res_frames (dai.Node.Output): Frames corresponding to the detection NN
            size (Tuple[int, int]): Size of the frames.
            debug (bool, optional): Enable debug mode. Defaults to False.
        """
        self.script: dai.node.Script = pipeline.create(dai.node.Script)
        self.script.setProcessor(dai.ProcessorType.LEON_CSS)  # More stable
        self._size: Tuple[int, int] = size

        detection_node.out.link(self.script.inputs['detections'])
        high_res_frames.link(self.script.inputs['frames'])

        self.configure(MultiStageConfig(debug))

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(size)
        self.manip.setWaitForConfigInput(True)
        self.manip.setMaxOutputFrameSize(size[0] * size[1] * 3)
        self.manip.setNumFramesPool(20)
        self.script.outputs['manip_cfg'].link(self.manip.inputConfig)
        self.script.outputs['manip_img'].link(self.manip.inputImage)
        self.out: dai.Node.Output = self.manip.out

    def configure(self, config: MultiStageConfig = None) -> None:
        """
        Args:
            config (MultiStageConfig, optional): Configuration object. Defaults to None.
        """
        if config is None:
            return

        with open(Path(os.path.dirname(__file__)) / 'template_multi_stage_script.py', 'r') as file:
            code = Template(file.read()).substitute(
                DEBUG='' if config.debug else '#',
                CHECK_LABELS=f"if det.label not in {str(config.labels)}: continue" if config.labels else "",
                WIDTH=str(self._size[0]),
                HEIGHT=str(self._size[1]),
                SCALE_BB_XMIN=f"-{config.scale_bb[0] / 100}" if config.scale_bb else "",  # % to float value
                SCALE_BB_YMIN=f"-{config.scale_bb[1] / 100}" if config.scale_bb else "",
                SCALE_BB_XMAX=f"+{config.scale_bb[0] / 100}" if config.scale_bb else "",
                SCALE_BB_YMAX=f"+{config.scale_bb[1] / 100}" if config.scale_bb else "",
            )
            self.script.setScript(code)
            # print(f"\n------------{code}\n---------------")

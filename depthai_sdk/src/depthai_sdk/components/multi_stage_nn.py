import os
from pathlib import Path
from string import Template
from typing import Tuple, Optional, List

import depthai as dai

from depthai_sdk.types import GenericNeuralNetwork


class MultiStageNN:
    def __init__(self,
                 pipeline: dai.Pipeline,
                 detection_node: GenericNeuralNetwork,  # Object detection node
                 high_res_frames: dai.Node.Output,
                 size: Tuple[int, int],
                 num_frames_pool: int = 20,
                 ) -> None:
        """
        Args:
            pipeline (dai.Pipeline): Pipeline object
            detection_node (GenericNeuralNetwork): Object detection NN
            high_res_frames (dai.Node.Output): Frames corresponding to the detection NN
            size (Tuple[int, int]): Size of the frames.
            debug (bool, optional): Enable debug mode. Defaults to False.
            num_frames_pool (int, optional): Number of frames to keep in the pool. Defaults to 20.
        """
        self.script: dai.node.Script = pipeline.create(dai.node.Script)
        self.script.setProcessor(dai.ProcessorType.LEON_CSS)  # More stable
        self._size: Tuple[int, int] = size

        detection_node.out.link(self.script.inputs['detections'])
        high_res_frames.link(self.script.inputs['frames'])

        self.configure() # User might later call this again with different parameters

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(size)
        self.manip.setWaitForConfigInput(True)
        self.manip.setMaxOutputFrameSize(size[0] * size[1] * 3)
        self.manip.setNumFramesPool(num_frames_pool)
        self.script.outputs['manip_cfg'].link(self.manip.inputConfig)
        self.script.outputs['manip_img'].link(self.manip.inputImage)
        self.out: dai.Node.Output = self.manip.out

    def configure(self,
                  debug: bool = False,
                  whitelist_labels: Optional[List[int]] = None,
                  scale_bb: Optional[Tuple[int, int]] = None) -> None:
        """
        Args:
            config (MultiStageConfig, optional): Configuration object. Defaults to None.
        """
        # Used later for visualization
        self.whitelist_labels = whitelist_labels
        self.scale_bb = scale_bb

        with open(Path(os.path.dirname(__file__)) / 'template_multi_stage_script.py', 'r') as file:
            code = Template(file.read()).substitute(
                DEBUG='' if debug else '#',
                CHECK_LABELS=f"if det.label not in {str(whitelist_labels)}: continue" if whitelist_labels else "",
                WIDTH=str(self._size[0]),
                HEIGHT=str(self._size[1]),
                SCALE_BB_XMIN=f"-{scale_bb[0] / 100}" if scale_bb else "",  # % to float value
                SCALE_BB_YMIN=f"-{scale_bb[1] / 100}" if scale_bb else "",
                SCALE_BB_XMAX=f"+{scale_bb[0] / 100}" if scale_bb else "",
                SCALE_BB_YMAX=f"+{scale_bb[1] / 100}" if scale_bb else "",
            )
            self.script.setScript(code)
            # print(f"\n------------{code}\n---------------")

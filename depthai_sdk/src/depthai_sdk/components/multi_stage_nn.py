import os
from pathlib import Path
from string import Template
from typing import Tuple, Optional, List
from depthai_sdk.classes.enum import ResizeMode
import depthai as dai

from depthai_sdk.types import GenericNeuralNetwork


class MultiStageNN:
    def __init__(self,
                 pipeline: dai.Pipeline,
                 detection_node: GenericNeuralNetwork,  # Object detection node
                 high_res_frames: dai.Node.Output,
                 size: Tuple[int, int],
                 frame_size: Tuple[int, int],
                 det_nn_size: Tuple[int, int],
                 resize_mode: ResizeMode,
                 num_frames_pool: int = 20,
                 ) -> None:
        """
        Args:
            pipeline (dai.Pipeline): Pipeline object
            detection_node (GenericNeuralNetwork): Object detection NN
            high_res_frames (dai.Node.Output): Frames corresponding to the detection NN
            size (Tuple[int, int]): NN input size of the second stage NN
            frame_size (Tuple[int, int]): Frame size of the first (detection) stage NN
            det_nn_size (Tuple[int, int]): NN input size of the first (detection) stage NN
            resize_mode (ResizeMode): Resize mode that was used to resize frame for first (detection) stage
            debug (bool, optional): Enable debug mode. Defaults to False.
            num_frames_pool (int, optional): Number of frames to keep in the pool. Defaults to 20.
        """
        frame_size_ar = frame_size[0] / frame_size[1]
        det_nn_size_ar = det_nn_size[0] / det_nn_size[1]
        if resize_mode == ResizeMode.LETTERBOX:
            padding = (frame_size_ar - det_nn_size_ar) / 2
            if padding > 0:
                self.init = f"xmin = 0; ymin = {-padding}; xmax = 1; ymax = {1 + padding}"
            else:
                self.init = f"xmin = {padding}; ymin = 0; xmax = {1 - padding}; ymax = 1"
        elif resize_mode in [ResizeMode.CROP, ResizeMode.FULL_CROP]:
            cropping = (1 - (det_nn_size_ar / frame_size_ar)) / 2
            if cropping < 0:
                self.init = f"xmin = 0; ymin = {-cropping}; xmax = 1; ymax = {1 + cropping}"
            else:
                self.init = f"xmin = {cropping}; ymin = 0; xmax = {1 - cropping}; ymax = 1"
        else: # Stretch
            self.init = f"xmin=0; ymin=0; xmax=1; ymax=1"

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
                INIT=self.init,
                SCALE_BB_XMIN=f"-{scale_bb[0] / 100}" if scale_bb else "",  # % to float value
                SCALE_BB_YMIN=f"-{scale_bb[1] / 100}" if scale_bb else "",
                SCALE_BB_XMAX=f"+{scale_bb[0] / 100}" if scale_bb else "",
                SCALE_BB_YMAX=f"+{scale_bb[1] / 100}" if scale_bb else "",
            )
            self.script.setScript(code)
            # print(f"\n------------{code}\n---------------")

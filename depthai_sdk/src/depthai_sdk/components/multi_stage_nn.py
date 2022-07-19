import depthai as dai
from typing import Tuple, Union, Optional, List
from string import Template

class MultiStageNN():
    script: dai.node.Script
    manip: dai.node.ImageManip
    out: dai.Node.Output # Cropped imgFrame output

    _size: Tuple[int, int]
    def __init__(self,
        pipeline: dai.Pipeline,
        detector: Union[
            dai.node.MobileNetDetectionNetwork,
            dai.node.MobileNetSpatialDetectionNetwork,
            dai.node.YoloDetectionNetwork,
            dai.node.YoloSpatialDetectionNetwork], # Object detector
        highResFrames: dai.Node.Output,
        size: Tuple[int, int],
        debug = False
        ) -> None:
        """
        Args:
            detections (dai.Node.Output): Object detection output
            highResFrames (dai.Node.Output): Output that will provide high resolution frames
        """

        self.script = pipeline.create(dai.node.Script)
        self._size = size

        detector.out.link(self.script.inputs['detections'])
        # Remove in 2.18 and use `imgFrame.getSequenceNum()` in Script node
        detector.passthrough.link(self.script.inputs['passthrough'])

        highResFrames.link(self.script.inputs['frames'])

        self.configMultiStageNn(debug = debug,)

        self.manip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(size)
        self.manip.setWaitForConfigInput(True)
        self.script.outputs['manip_cfg'].link(self.manip.inputConfig)
        self.script.outputs['manip_img'].link(self.manip.inputImage)

    def configMultiStageNn(self,
        debug = False,
        labels: Optional[List[int]] = None,
        ) -> None:
        # TODO: add support to scale bounding box
        with open('template_multi_stage_script.py', 'r') as file:
            code = Template(file.read()).substitute(
                DEBUG = '' if debug else '#',
                CHECK_LABELS = f"if det.label not in {str(labels)}: continue" if labels else "",
                WIDTH = str(self._size[0]),
                HEIGHT = str(self._size[1]),
            )
            print('Multi stage script code:', code)
            self.script.setScript(code)


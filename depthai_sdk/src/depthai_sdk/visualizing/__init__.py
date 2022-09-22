from typing import Tuple, Union, Dict, Callable
from .visualizers import BaseVisualizer, DetectionVisualizer, DetectionClassificationVisualizer, FPS, FrameVisualizer, \
    FramePosition, SpatialBbMappingsVisualizer, DisparityVisualizer, DepthVisualizer
from .visualizer_helper import Visualizer
from ..classes.xout import XoutNnResults, XoutFrames, XoutTwoStage, XoutSpatialBbMappings, XoutDisparity, XoutDepth


class VisualizerManager:
    """
    VisualizerManager will create required visualizers on setup() based on components we want to visualize.
    Each VisualizerManager can contain multiple visualizers.
    """
    _visualizer: BaseVisualizer = None
    _scale: Union[None, float, Tuple[int, int]]
    _fps: Dict[str, FPS]
    _callback: Callable

    def __init__(self,
                 scale: Union[None, float, Tuple[int, int]] = None,
                 fps: Dict[str, FPS] = None,
                 callback: Callable = None):
        """
        Create the VisualizerManager.

        @param scale: Optionally rescale the frame
        @param fpsHandlers: FPS counters if we want to display FPS on the frame
        @param callback: Callback to be called instead of displaying the frame. Synced packet will be sent to the callback
        """
        self._scale = scale
        self._fps = fps
        self._callback = callback

    def setup(self, xout) -> None:
        """
        Called after connected to the device and all components have been configured.
        """

        if isinstance(xout, XoutNnResults):
            self._visualizer = DetectionVisualizer(xout)

        elif isinstance(xout, XoutDisparity):
            # Needs to be before checking if instanceof XoutFrames
            self._visualizer = DisparityVisualizer(xout)

        elif isinstance(xout, XoutDepth):
            # Needs to be before checking if instanceof XoutFrames
            self._visualizer = DepthVisualizer(xout.frames.name)

        elif isinstance(xout, XoutFrames):
            self._visualizer = BaseVisualizer(xout.frames.name)
        # elif group.type == GroupType.FRAMES:
        #     vis = BaseVisualizer(frame_name)
        # elif group.type == GroupType.LOG:
        #     # TODO: We don't have any frame to display NN results on, so just print NN results
        #     raise NotImplementedError('Printing NN results not yet implemented')
        # elif group.type == GroupType.DETECTIONS:
        #     # We have one detector
        #     vis =
        elif isinstance(xout, XoutTwoStage):
            self._visualizer = DetectionClassificationVisualizer(xout)
        # # TODO: if classification network, display NN results in the bounding box
        elif isinstance(xout, XoutSpatialBbMappings):
            self._visualizer = SpatialBbMappingsVisualizer(xout)
        else:
            raise NotImplementedError('Visualization of these components is not yet implemented!')

        self._visualizer.setBase(self._scale, self._fps, self._callback)

    # Called via callback
    def new_msgs(self, msgs: Dict):
        self._visualizer.newMsgs(msgs)

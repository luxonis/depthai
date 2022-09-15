from typing import Tuple, Union, List, Dict, Callable
from ..components import Component, GroupType, ComponentGroup
from .visualizers import BaseVisualizer, DetectionVisualizer, DetectionClassificationVisualizer, FPS, FrameVisualizer
from .visualizer_helper import Visualizer
from .packets import TwoStagePacket, DetectionPacket, FramePacket

class VisualizerManager:
    """
    VisualizerManager will create required visualizers on setup() based on components we want to visualize.
    Each VisualizerManager can contain multiple visualizers.
    """
    _visualizers: List[BaseVisualizer] = []
    _scale: Union[None, float, Tuple[int, int]] = None
    _fps: Dict[str, FPS] = None
    _callback: Callable = None
    _components: List[Component]

    def __init__(self, components: List[Component],
                 scale: Union[None, float, Tuple[int, int]] = None,
                 fps: Dict[str, FPS] = None,
                 callback: Callable = None):
        """
        Create the VisualizerManager.

        @param components: Components which we want to visualize
        @param scale: Optionally rescale the frame
        @param fpsHandlers: FPS counters if we want to display FPS on the frame
        @param callback: Callback to be called instead of displaying the frame. Synced packet will be sent to the callback
        """
        self._scale = scale
        self._fps = fps
        self._callback = callback
        self._components = components

    def setup(self) -> None:
        """
        Called after connected to the device and all components have been configured.
        """
        group = ComponentGroup(self._components)

        for frame_name in group.frame_names:
            vis: BaseVisualizer
            # We only have frames to display, no NN results
            if frame_name.startswith('__'):
                vis = FrameVisualizer(frame_name)
            elif group.type == GroupType.FRAMES:
                vis = BaseVisualizer(frame_name)
            elif group.type == GroupType.LOG:
                # TODO: We don't have any frame to display NN results on, so just print NN results
                raise NotImplementedError('Printing NN results not yet implemented')
            elif group.type == GroupType.DETECTIONS:
                # We have one detector
                vis = DetectionVisualizer(frame_name, group.nn_name, group.nn_component)
            elif group.type == GroupType.MULTI_STAGE:
                vis = DetectionClassificationVisualizer(frame_name, group)
            # TODO: if classification network, display NN results in the bounding box
            else:
                raise NotImplementedError('Visualization of these components is not yet implemented!')

            vis.setBase(self._scale, self._fps, self._callback)
            self._visualizers.append(vis)

    # Called via callback
    def new_msgs(self, msgs: Dict):
        for vis in self._visualizers:
            vis.newMsgs(msgs)

import depthai as dai
from typing import Tuple, Union, List, Dict, Type, Callable
from ..components import Component, NNComponent
from .visualizers import BaseVisualizer, DetectionVisualizer, FramePosition, FPS


class Visualizer:
    _components: List[Component]
    _visualizers: List[BaseVisualizer] = []
    _scale: Union[None, float, Tuple[int, int]] = None
    _fps: Dict[str, FPS] = None
    _callback: Callable = None

    def __init__(self, components: List[Component],
                 scale: Union[None, float, Tuple[int, int]] = None,
                 fpsHandlers: Dict[str, FPS] = None,
                 callback: Callable = None) -> None:
        self._components = components
        self._scale = scale
        self._fps = fpsHandlers
        self._callback = callback

    def setup(self):
        """
        Called after connected to the device, and all components have been configured
        @return:
        """
        detectors = self._get_obj_detectors(self._components)
        nonDetectors = self._get_non_obj_detectors(self._components)
        frames = self._streams_by_type(self._components, dai.ImgFrame)

        if len(detectors) == 0 and len(nonDetectors) == 0:
            # We only have frames to display, no NN results
            for frame in frames:
                vis = BaseVisualizer(frame)
                vis.setBase(self._scale, self._fps, self._callback)
                self._visualizers.append(vis)
        elif len(frames) == 0:
            # TODO: We don't have any frame to display NN results on, so just print NN results
            raise NotImplementedError('Printing NN results not yet implemented')
        elif len(detectors) == 1 and len(nonDetectors) == 0:
            # We have one detector
            nn = detectors[0]
            for frame in frames:
                nnStreamNames = self._streams_by_type_xout(nn.xouts, dai.ImgDetections)
                detVis = DetectionVisualizer(frame, nnStreamNames[0], nn)
                detVis.setBase(self._scale, self._fps, self._callback)
                self._visualizers.append(detVis)
        elif len(detectors) == 1 and len(nonDetectors) == 1:
            # 1 detector + 1 non-detector
            nn = detectors[0]
            for frame in frames:
                nnStreamNames = self._streams_by_type_xout(nn.xouts, dai.ImgDetections)
                detVis = DetectionVisualizer(frame, nnStreamNames[0], nn)
                detVis.setBase(self._scale, self._fps, self._callback)
                self._visualizers.append(detVis)
            # TODO: if classification network, display NN results in the bouding box
        else:
            raise NotImplementedError('Visualization of these components is not yet implemented!')

    def _get_stream_name(self, xouts: Dict, type: Type) -> str:
        for name, (compType, daiType) in xouts.items():
            if daiType == type: return name
        raise ValueError('Stream name was not found in these Xouts!')

    # Called via callback
    def new_msgs(self, msgs: Dict):
        for vis in self._visualizers:
            vis.newMsgs(msgs)
        # frame = self._getFirstMsg(msgs, dai.ImgFrame).getCvFrame()
        # dets = self._getFirstMsg(msgs, dai.ImgDetections).detections

    def _msgs_list(self, msgs: Dict) -> List[Tuple]:
        arr = []
        for name, msg in msgs:
            arr.append((name, msg, type(msg)))
        return arr

    def _streams_by_type(self, components: List[Component], type: Type) -> List[str]:
        streams = []
        for comp in components:
            for name, (compType, daiType) in comp.xouts.items():
                if daiType == type:
                    streams.append(name)
        return streams

    def _components_by_type(self, components: List[Component], type: Type) -> List[Component]:
        comps = []
        for comp in components:
            if isinstance(comp, type):
                comps.append(comp)
        return comps

    def _getTypeDict(self, msgs: Dict) -> Dict[str, Type]:
        ret = dict()
        for name, msg in msgs:
            ret[name] = type(msg)
        return ret

    def _get_obj_detectors(self, components: List[Component]) -> List[Component]:
        """
        Finds object detection NNComponents from the list of all components
        """
        comps = []
        for comp in components:
            if isinstance(comp, NNComponent) and comp.isDetector():
                comps.append(comp)
        return comps

    def _get_non_obj_detectors(self, components: List[Component]) -> List[Component]:
        """
        Reverse function of the `_get_obj_decetors`
        """
        comps = []
        for comp in components:
            if isinstance(comp, NNComponent) and not comp.isDetector():
                comps.append(comp)
        return comps

    def _get_component(self, streamName: str) -> Component:
        for comp in self._components:
            if streamName in comp.xouts:
                return comp

        raise ValueError("[SDK Visualizer] stream name wasn't found in any component!")
        return

    def _streams_by_type_xout(self, xouts: Dict[str, Tuple[type, type]], type: Type) -> List[str]:
        streams = []
        for name, (compType, daiType) in xouts.items():
            if type == daiType:
                streams.append(name)
        return streams
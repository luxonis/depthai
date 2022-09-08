from enum import IntEnum
from .component import Component
from typing import List, Dict, Tuple, Type
from ..components.nn_component import NNComponent
import depthai as dai

class GroupType(IntEnum):
    OTHER=0 # Unsupported group type; NoSync, print (non-)synced messages
    FRAMES=1 # If only frames we don't need syncing (NoSync) and we can use BaseVisualizer
    DETECTIONS=2 # Frames+detections. Use SequenceSync for now (later: TimestampSync)
    TWO_STAGE=3 # Frames + detections + another NN. Use TwoStageSeqSync (later: timestap-based Two-Stage syncing)



class ComponentGroup:
    """
    This class contains a group of components that we want to use together, and helps with syncing
    and visualization.
    """
    group_type: GroupType

    frames: List[str]

    det_name: str
    det_component: NNComponent

    recognition_name: str
    recognition_component: NNComponent

    def __init__(self, components: List[Component]):
        detectors = _get_obj_detectors(components)
        recognition_nns = _get_recognition_nns(components)
        self.frames = _streams_by_type(components, dai.ImgFrame)

        if len(detectors) == 0 and len(recognition_nns) == 0:
            # We only have frames to display, no NN results
            self.group_type = GroupType.FRAMES
        elif len(self.frames) == 0:
            # TODO: We don't have any frame to display NN results on, so just print NN results
            raise NotImplementedError('Printing NN results not yet implemented')
            self.group_type = GroupType.OTHER
        elif len(detectors) == 1 and len(recognition_nns) == 0:
            self.group_type = GroupType.DETECTIONS
            self.det_component = detectors[0]
            nnStreamNames = _streams_by_type_xout(self.det_component.xouts, dai.ImgDetections)
            self.det_name = nnStreamNames[0]
        elif len(detectors) == 1 and len(recognition_nns) == 1:
            self.group_type = GroupType.TWO_STAGE

            self.det_component = detectors[0]
            nnStreamNames = _streams_by_type_xout(self.det_component.xouts, dai.ImgDetections)
            self.det_name = nnStreamNames[0]

            self.recognition_component = recognition_nns[0]
            recNames = _streams_by_type_xout(self.det_component.xouts, dai.NNData)
            self.recognition_name = recNames[0]


        else:
            raise NotImplementedError('Visualization of these components is not yet implemented!')


def _get_stream_name(xouts: Dict, type: Type) -> str:
    for name, (compType, daiType) in xouts.items():
        if daiType == type: return name
    raise ValueError('Stream name was not found in these Xouts!')

def _msgs_list(msgs: Dict) -> List[Tuple]:
    arr = []
    for name, msg in msgs:
        arr.append((name, msg, type(msg)))
    return arr

def _streams_by_type(components: List[Component], type: Type) -> List[str]:
    streams = []
    for comp in components:
        for name, (compType, daiType) in comp.xouts.items():
            if daiType == type:
                streams.append(name)
    return streams

def _components_by_type(components: List[Component], type: Type) -> List[Component]:
    comps: List[Component] = []
    for comp in components:
        if isinstance(comp, type):
            comps.append(comp)
    return comps

def _get_type_dict(msgs: Dict) -> Dict[str, Type]:
    ret = dict()
    for name, msg in msgs:
        ret[name] = type(msg)
    return ret

def _get_obj_detectors(components: List[Component]) -> List[NNComponent]:
    """
    Finds object detection NNComponents from the list of all components
    """
    comps = []
    for comp in components:
        if isinstance(comp, NNComponent) and comp.isDetector():
            comps.append(comp)
    return comps

def _get_recognition_nns(components: List[Component]) -> List[NNComponent]:
    """
    Reverse function of the `_get_obj_decetors`
    """
    comps = []
    for comp in components:
        if isinstance(comp, NNComponent) and not comp.isDetector():
            comps.append(comp)
    return comps

def _streams_by_type_xout(xouts: Dict[str, Tuple[type, type]], type: Type) -> List[str]:
    streams = []
    for name, (compType, daiType) in xouts.items():
        if type == daiType:
            streams.append(name)
    return streams
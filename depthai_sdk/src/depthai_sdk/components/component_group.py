from enum import IntEnum
from typing import List, Dict, Tuple, Type, Union
from ..components import CameraComponent, StereoComponent, NNComponent, Component
import depthai as dai


class GroupType(IntEnum):
    LOG = 0  # Unsupported group type; NoSync, print (non-)synced messages
    FRAMES = 1  # If only frames we don't need syncing (NoSync) and we can use BaseVisualizer
    RECOGNITION = 2 # Frames + recognition NN results. Use SequenceSync
    DETECTIONS = 3  # Frames+detections. Use SequenceSync for now (later: TimestampSync)
    MULTI_STAGE = 4  # Frames + detections + another NN. Use TwoStageSeqSync (later: timestamp-based Two-Stage syncing)


"""
What kind of dataflow we want to synchronize / visualize
"""


class ComponentGroup:
    """
    This class contains a group of components that we want to use together, and helps with syncing
    and visualization.
    """
    type: GroupType
    components: List[Component]

    frame_component: Union[CameraComponent, StereoComponent]
    frame_names: List[str]

    nn_component: NNComponent
    nn_name: str  # Stream name

    second_nn: NNComponent
    second_nn_name: str  # Stream name


    def __init__(self, components: List[Component]):
        """
        Resolves what kind of data flow we have with multiple nodes, which helps with syncing and visualization.
        @param components: List of Components
        """
        self.components = components

        detection_nns = _object_detectors(components)
        recognition_nns = _recognition_nns(components)
        second_stage_nns = _second_stage_nn(components)
        frames = _streams_by_type(components, dai.ImgFrame)

        if len(frames) == 0:
            # We don't have any frame to display things on, just log all messages
            self.type = GroupType.LOG
            return

        self.frame_names = frames

        if len(detection_nns) == 0 and len(recognition_nns) == 0:
            # We only have frames to display, no NN results
            self.type = GroupType.FRAMES
            return

        if len(detection_nns) == 0 and len(recognition_nns) != 0:
            # We only have recognition non-detection NN results
            self.type = GroupType.RECOGNITION
            return


        if len(detection_nns) != 0:
            # We have some detection results
            self.type = GroupType.DETECTIONS

            self.nn_component = detection_nns[0]
            self.nn_name = _streams_by_type_xout(self.nn_component.xouts, dai.ImgDetections)[0]

            if len(second_stage_nns) == 1:
                self.type = GroupType.MULTI_STAGE

                self.second_nn = second_stage_nns[0]
                self.second_nn_name = _streams_by_type_xout(self.second_nn.xouts, dai.NNData)[0]


def _object_detectors(components: List[Component]) -> List[NNComponent]:
    """
    Finds object detection NNComponents from the list of all components
    """
    comps = []
    for comp in components:
        if isinstance(comp, NNComponent) and comp.isDetector():
            comps.append(comp)
    return comps


def _recognition_nns(components: List[Component]) -> List[NNComponent]:
    """
    Reverse function of the `_get_obj_detectors`
    """
    comps = []
    for comp in components:
        if isinstance(comp, NNComponent) and not comp.isDetector():
            comps.append(comp)
    return comps


def _second_stage_nn(components: List[Component]) -> List[NNComponent]:
    comps = []
    for comp in components:
        if isinstance(comp, NNComponent) and isinstance(comp._input, NNComponent):
            comps.append(comp)
    return comps


def _streams_by_type_xout(xouts: Dict[str, Tuple[type, type]], msg_type: Type) -> List[str]:
    streams = []
    for name, (compType, daiType) in xouts.items():
        if msg_type == daiType:
            streams.append(name)
    return streams


def _streams_by_type(components: List[Component], msg_type: Type) -> List[str]:
    streams = []
    for comp in components:
        for name, (compType, daiType) in comp.xouts.items():
            if msg_type == daiType:
                streams.append(name)
    return streams

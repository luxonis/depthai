from datetime import timedelta
from typing import Dict, Union

import depthai as dai

from depthai_sdk import DetectionPacket, NNComponent
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger


class DetectionTrigger(Trigger):
    def __init__(self,
                 input: NNComponent,
                 min_detections: Dict[str, int],
                 cooldown: Union[timedelta, int]):
        if not isinstance(input.node, dai.node.DetectionNetwork):
            raise ValueError("The input NNComponent must represent a node of type 'DetectionNetwork'")
        for label, number in min_detections.items():
            if label not in input.get_labels():
                raise ValueError(f"Label '{label}' doesn't exist for network '{input.get_name()}'. "
                                 f"Supported labels are:\n {input.get_labels()}")
            if number <= 0:
                raise ValueError("Numbers given in min_detections must be positive")
        self.labels = input.get_labels()
        self.min_detections = min_detections
        super().__init__(input.out.main, self.condition, cooldown)

    def condition(self, packet: DetectionPacket) -> bool:
        dets = packet.img_detections.detections
        dets_of_interest = dict.fromkeys(self.min_detections.keys(), 0)
        for det in dets:
            label_str = self.labels[det.label]
            if dets_of_interest.get(label_str) is not None:
                dets_of_interest[label_str] += 1
        return all(a >= b for a, b in zip(list(dets_of_interest.values()), list(self.min_detections.values())))

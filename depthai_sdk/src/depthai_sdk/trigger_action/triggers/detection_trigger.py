from datetime import timedelta
from typing import Dict, Union

import depthai as dai

from depthai_sdk.classes import DetectionPacket
from depthai_sdk.components import NNComponent
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger

__all__ = ['DetectionTrigger']


class DetectionTrigger(Trigger):
    """
    Trigger that is activated when a certain number of detections of a certain label are detected.
    """

    def __init__(self,
                 input: NNComponent,
                 min_detections: Dict[str, int],
                 cooldown: Union[timedelta, int]):
        """
        Args:
            input: NNComponent that represents the NN node that will be used for detection.
            min_detections: Dictionary that maps labels to the minimum number of detections of that label that must be
                            detected for the trigger to be activated. Example: `{'person': 1, 'dog': 2}` means that
                            the trigger will be activated when at least 1 person and 2 dogs are detected.
            cooldown: Time that must pass after the trigger is activated before it can be activated again.
        """
        if not isinstance(input.node, dai.node.DetectionNetwork):
            raise ValueError('The input NNComponent must represent a node of type `dai.node.DetectionNetwork`')

        self.min_detections = min_detections
        # Convert all labels to uppercase
        self.min_detections = {label.upper(): number for label, number in min_detections.items()}

        for label, number in self.min_detections.items():
            if label not in input.get_labels():
                raise ValueError(f'Label "{label}" doesn\'t exist for network "{input.get_name()}". '
                                 f'Supported labels are:\n {input.get_labels()}')

            if number <= 0:
                raise ValueError('Numbers given in min_detections must be positive.')

        self.labels = input.get_labels()
        super().__init__(input.out.main, self.condition, cooldown)

    def condition(self, packet: DetectionPacket) -> bool:
        """
        Method that checks if the trigger should be activated.

        Args:
            packet: DetectionPacket that contains the detections.

        Returns:
            True if the trigger should be activated, False otherwise.
        """
        dets = packet.img_detections.detections

        dets_of_interest = dict.fromkeys(self.min_detections.keys(), 0)
        for det in dets:
            label_str = self.labels[det.label]
            if dets_of_interest.get(label_str) is not None:
                dets_of_interest[label_str] += 1

        return all(a >= b for a, b in zip(list(dets_of_interest.values()), list(self.min_detections.values())))

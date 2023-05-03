from abc import ABC
from typing import Callable, Union, List, Dict, Optional

from depthai_sdk.classes import FramePacket
from depthai_sdk.components import Component

__all__ = ['Action']


class Action(ABC):
    """
    Base action represents a single action that can be activated by a trigger.
    """

    def __init__(self,
                 inputs: Optional[Union[Component, Callable, List[Union[Component, Callable]]]] = None,
                 action: Optional[Callable] = None):
        if inputs:
            if not isinstance(inputs, list):
                inputs = [inputs]

            for i in range(len(inputs)):
                if isinstance(inputs[i], Component):
                    inputs[i] = inputs[i].out.main

        self.inputs = inputs
        self.stream_names: List[str] = []
        self.action = action

    def activate(self) -> None:
        """
        Method that gets called when the action is activated by a trigger.

        Returns:
            None.
        """
        if self.action:
            self.action()

    def on_new_packets(self, packets: Dict[str, FramePacket]) -> None:
        """
        Callback method that gets called when all packets are synced.

        Args:
            packets: Dictionary of packets received from the input streams.

        Returns:
            None.
        """
        pass

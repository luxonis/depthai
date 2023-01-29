from typing import Callable, Union, List, Dict, Optional

from depthai_sdk import Component, FramePacket


class Action:
    def __init__(self,
                 inputs: Optional[Union[Component, Callable, List[Union[Component, Callable]]]] = None,
                 action: Optional[Callable] = None):
        if inputs:
            if isinstance(inputs, Component):
                inputs = inputs.out.main

            if isinstance(inputs, Callable):
                inputs = [inputs]

            for i in range(len(inputs)):
                if isinstance(inputs[i], Component):
                    inputs[i] = inputs[i].out.main

        self.inputs = inputs
        self.stream_names: List[str] = []
        self.action = action

    def after_trigger(self):
        if self.action:
            self.action()

    def process_packets(self, packets: Dict[str, FramePacket]):
        pass

    def run_thread(self):
        pass

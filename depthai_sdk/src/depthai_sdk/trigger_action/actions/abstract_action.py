from abc import ABC, abstractmethod
from typing import Callable, Union

from depthai_sdk import Component


class Action(ABC):
    def __init__(self, input: Union[Component, Callable]):  # extend to List[Callable] and add sync
        if isinstance(input, Component):
            input = input.out.main
        self.input = input

    @abstractmethod
    def action(self):
        raise NotImplementedError()

from abc import ABC
from datetime import timedelta
from typing import Union, Callable

from depthai_sdk import Component


class Trigger(ABC):
    def __init__(self, input: Union[Component, Callable], condition: Callable, period: int):
        if isinstance(input, Component):
            input = input.out.main
        self.input = input
        self.condition = condition
        self.period = timedelta(seconds=period)

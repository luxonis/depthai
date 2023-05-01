from abc import ABC
from datetime import timedelta
from typing import Union, Callable

from depthai_sdk import Component


class Trigger(ABC):
    def __init__(self, input: Union[Component, Callable], condition: Callable, cooldown: Union[timedelta, int]):
        if isinstance(input, Component):
            input = input.out.main
        self.input = input
        self.condition = condition
        if isinstance(cooldown, timedelta):
            cooldown = cooldown.total_seconds()
        if cooldown >= 0:
            self.cooldown = timedelta(seconds=cooldown)
        else:
            raise ValueError("Cooldown time must be a non-negative integer or "
                             "a timedelta object representing non-negative time difference")
        self.cooldown = timedelta(seconds=cooldown)


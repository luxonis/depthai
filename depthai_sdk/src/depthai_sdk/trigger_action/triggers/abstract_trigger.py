from abc import ABC
from datetime import timedelta
from typing import Union, Callable

from depthai_sdk.components import Component

__all__ = ['Trigger']


class Trigger(ABC):
    """
    Base trigger represents a single trigger that can activate an action.
    """

    def __init__(self, input: Union[Component, Callable], condition: Callable, cooldown: Union[timedelta, int]):
        """
        Args:
            input: Input component or output of a component that will be used as input for the trigger.
            condition: Condition that will be used to check if the trigger should be activated.
            cooldown: Cooldown time in seconds. If the trigger is activated, it will be ignored for the next `cooldown` seconds.
        """
        if isinstance(input, Component):
            input = input.out.main

        if isinstance(cooldown, timedelta):
            cooldown = cooldown.total_seconds()

        if cooldown >= 0:
            self.cooldown = timedelta(seconds=cooldown)
        else:
            raise ValueError("Cooldown time must be a non-negative integer or "
                             "a timedelta object representing non-negative time difference")

        self.input = input
        self.condition = condition
        self.cooldown = timedelta(seconds=cooldown)

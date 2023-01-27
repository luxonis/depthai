from datetime import timedelta
from pathlib import Path
from typing import Union, Callable, Tuple

from depthai_sdk import Component
from depthai_sdk.trigger_action.actions.abstract_action import Action


class RecordAction(Action):
    def __init__(self,
                 input: Union[Component, Callable],
                 path: str,
                 duration_before_trigger: Union[int, timedelta],
                 duration_after_trigger: Union[timedelta, int]
                 ):
        super().__init__(input)
        self.path = Path(path).resolve()
        if isinstance(duration_before_trigger, timedelta):
            duration_before_trigger = duration_before_trigger.total_seconds()
        if isinstance(duration_after_trigger, timedelta):
            duration_after_trigger = duration_after_trigger.total_seconds()
        if duration_before_trigger > 0 and duration_after_trigger > 0:
            self.duration_before_trigger = duration_before_trigger
            self.duration_after_trigger = duration_after_trigger
        else:
            raise ValueError("Recording durations before and after trigger must be positive integers "
                             "or positive timedelta objects")

    def action(self):
        pass  # Is done in RecordController, probably needs to be changed...

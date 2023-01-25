from pathlib import Path
from typing import Union, Callable, Tuple

from depthai_sdk import Component
from depthai_sdk.trigger_action.actions.abstract_action import Action


class RecordAction(Action):
    path: Path
    duration: Tuple[int, int]

    def __init__(self, input: Union[Component, Callable], path: str, duration: Tuple[int, int]):
        super().__init__(input)
        self.path = Path(path).resolve()
        self.duration = duration

    def action(self):
        pass

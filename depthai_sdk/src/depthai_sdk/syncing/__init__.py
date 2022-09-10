from typing import List, Callable
from .syncing import BaseSync, NoSync, SequenceSync, TwoStageSeqSync
from ..components import Component, ComponentGroup, GroupType


class Sync:
    components: List[Component]
    callback: Callable
    base: BaseSync

    def __init__(self, components: List[Component], callback: Callable):
        self.components = components
        self.callback = callback

    def setup(self):
        group = ComponentGroup(self.components)

        if group.type == GroupType.DETECTIONS:
            self.base = SequenceSync([self.callback], self.components)
        elif group.type == GroupType.MULTI_STAGE:
            self.base = TwoStageSeqSync([self.callback], group)
        else:
            self.base = NoSync([self.callback], self.components)

        self.base.setup()


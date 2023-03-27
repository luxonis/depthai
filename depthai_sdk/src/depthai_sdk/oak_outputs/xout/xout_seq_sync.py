from abc import abstractmethod
from typing import List

from depthai_sdk.oak_outputs.syncing import MessageSync
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout


class XoutMessageSync(XoutBase, MessageSync):
    def xstreams(self) -> List[StreamXout]:
        return self.streams

    def __init__(self, streams: List[StreamXout]):
        self.streams = streams
        # Save StreamXout before initializing super()!
        XoutBase.__init__(self)
        MessageSync.__init__(self, len(streams))
        self.msgs = dict()

    @abstractmethod
    def package(self, msgs: List):
        raise NotImplementedError('XoutMessageSync is an abstract class, you need to override package() method!')

    def new_msg(self, name: str, msg) -> None:
        # Ignore frames that we aren't listening for
        if name not in self._streams: return

        synced = self.sync(msg.getSequenceNum(), name, msg)
        if synced:
            self.package(synced)

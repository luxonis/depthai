from typing import Union, List, Callable
from queue import Empty, Queue
from abc import ABC, abstractmethod
import depthai as dai

class StreamXout:
    stream: dai.Node.Output
    name: str # XLinkOut stream name
    def __init__(self, id: int, out: dai.Node.Output):
        self.stream = out
        self.name = f"{str(id)}_{out.name}"

class ReplayStream(StreamXout):
    def __init__(self, name: str):
        self.name = name


class XoutBase(ABC):
    callbacks: List[Callable]  # List of callback to be called when we have new synced msgs
    queue: Queue  # Queue to which (synced/non-synced) messages will be added
    _streams: List[str]  # Streams to listen for

    @abstractmethod
    def xstreams(self) -> List[StreamXout]:
        raise NotImplementedError()

    def __init__(self) -> None:
        self._streams = [xout.name for xout in self.xstreams()]

    def setup_base(self, callback: Union[Callable, List[Callable]]):
        # Gets called when initializing
        self.queue = Queue(maxsize=30)
        if isinstance(callback, List):
            self.callbacks = callback
        else:
            self.callbacks = [callback]

    @abstractmethod
    def newMsg(self, name: str, msg) -> None:
        raise NotImplementedError()

    # This approach is used as some functions (eg. imshow()) need to be called from
    # main thread, and calling them from callback thread wouldn't work.
    def checkQueue(self, block=False) -> None:
        """
        Checks queue for any available messages. If available, call callback. Non-blocking by default.
        """
        try:
            msgs = self.queue.get(block=block)
            if msgs is not None:
                for cb in self.callbacks:  # Call all callbacks
                    cb(msgs)
        except Empty:  # Queue empty
            pass
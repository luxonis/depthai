import warnings
from abc import ABC, abstractmethod
from queue import Empty, Queue
from typing import List, Callable, Optional

import depthai as dai

from depthai_sdk.oak_outputs.fps import FPS


class StreamXout:
    def __init__(self, id: int, out: dai.Node.Output, name: Optional[str] = None):
        self.stream = out
        if name is not None:
            self.name = f'{name}_{str(out.name)}'
        else:
            self.name = f"{str(id)}_{out.name}"


class ReplayStream(StreamXout):
    def __init__(self, name: str):
        self.name = name


class XoutBase(ABC):
    def __init__(self) -> None:
        self._streams = [xout.name for xout in self.xstreams()]
        self._visualizer = None
        self._visualizer_enabled = False
        self._packet_name = None
        self._fps = None
        self.queue = None
        self.callback = None

    def get_packet_name(self) -> str:
        if self._packet_name is None:
            self._packet_name = ";".join([xout.name for xout in self.xstreams()])
        return self._packet_name

    @abstractmethod
    def xstreams(self) -> List[StreamXout]:
        raise NotImplementedError()

    def setup_base(self, callback: Callable):
        # Gets called when initializing
        self.queue = Queue(maxsize=10)
        self.callback = callback

    def start_fps(self):
        self._fps = FPS()

    @abstractmethod
    def new_msg(self, name: str, msg) -> None:
        raise NotImplementedError()

    @abstractmethod
    def visualize(self, packet) -> None:
        raise NotImplementedError()

    def on_callback(self, packet) -> None:
        """
        Hook called when `callback` or `self.visualize` are used.
        """
        pass

    def on_record(self, packet) -> None:
        """
        Hook called when `record_path` is used.
        """
        pass

    # This approach is used as some functions (eg. imshow()) need to be called from
    # main thread, and calling them from callback thread wouldn't work.
    def check_queue(self, block=False) -> None:
        """
        Checks queue for any available messages. If available, call callback. Non-blocking by default.
        """
        try:
            packet = self.queue.get(block=block)

            if packet is not None:
                self._fps.next_iter()

                self.on_callback(packet)

                if self._visualizer_enabled:
                    try:
                        self._visualizer.frame_shape = packet.frame.shape
                    except AttributeError:
                        pass  # Not all packets have frame attribute

                    if self._visualizer:
                        try:
                            self.visualize(packet)
                        except AttributeError:
                            warnings.warn('OpenCV (or another libraries) may not be installed, cannot visualize frames')
                else:
                    # User defined callback
                    self.callback(packet)

                # Record after processing, so that user can modify the frame
                self.on_record(packet)

        except Empty:  # Queue empty
            pass

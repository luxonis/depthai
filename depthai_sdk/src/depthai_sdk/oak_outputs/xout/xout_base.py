import traceback
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

    def close(self) -> None:
        """
        Hook that will be called when exiting the context manager.
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
                        self.visualize(packet)
                    except Exception as e:
                        warnings.warn(f'An error occurred while visualizing: {e}')
                        traceback.print_exc()
                else:
                    # User defined callback
                    try:
                        self.callback(packet)
                    except Exception as e:
                        warnings.warn(f'An error occurred while calling callback: {e}')
                        traceback.print_exc()

                # Record after processing, so that user can modify the frame
                self.on_record(packet)

        except Empty:  # Queue empty
            pass

    def fourcc(self) -> str:
        if self.is_mjpeg():
            return 'mjpeg'
        elif self.is_h264():
            return 'h264'
        elif self.is_h265():
            return 'hevc'
        elif self.is_depth():
            return 'y16'
        # TODO: add for mono, rgb, nv12, yuv...
        else:
            return None

    def is_h265(self) -> bool:
        if type(self).__name__ == 'XoutH26x':
            # XoutH26x class has profile attribute
            return self.profile == dai.VideoEncoderProperties.Profile.H265_MAIN
        return False

    def is_h264(self) -> bool:
        if type(self).__name__ == 'XoutH26x':
            # XoutH26x class has profile attribute
            return self.profile != dai.VideoEncoderProperties.Profile.H265_MAIN
        return False

    def is_h26x(self) -> bool:
        return type(self).__name__ == 'XoutH26x'

    def is_mjpeg(self) -> bool:
        return type(self).__name__ == 'XoutMjpeg'

    def is_raw(self) -> bool:
        return type(self).__name__ == 'XoutFrames'

    def is_depth(self) -> bool:
        return type(self).__name__ == 'XoutDepth'

    def is_disparity(self) -> bool:
        return type(self).__name__ == 'XoutDisparity'

    def is_imu(self):
        return type(self).__name__ == 'XoutIMU'

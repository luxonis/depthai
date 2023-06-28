from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Optional, Callable
import depthai as dai


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
        self._packet_name = None

        # self._fps_counter = dict()
        # for name in self._streams:
        #     self._fps_counter[name] = FPS()

        # It will get assigned later inside the BasePacketHandler class
        self.new_packet_callback: Callable

    def get_packet_name(self) -> str:
        if self._packet_name is None:
            self._packet_name = ";".join([xout.name for xout in self.xstreams()])
        return self._packet_name

    @abstractmethod
    def xstreams(self) -> List[StreamXout]:
        raise NotImplementedError()

    def device_msg_callback(self, name, dai_message) -> None:
        """
        This is the (first) callback that gets called on a device message. Don't override it.
        It will call `new_msg` and `on_callback` methods. If `new_msg` returns a packet, it will call
        `new_packet` method.
        """
        # self._fps_counter[name].next_iter()
        packet = self.new_msg(name, dai_message)
        if packet is not None:
            # If not list, convert to list.
            # Some Xouts create multiple packets from a single message (example: IMU)
            if not isinstance(packet, list):
                packet = [packet]

            for p in packet:
                self.on_callback(p)
                self.new_packet_callback(p)

    @abstractmethod
    def new_msg(self, name: str, msg) -> None:
        raise NotImplementedError()

    def on_callback(self, packet) -> None:
        """
        Hook called when `callback` or `self.visualize` are used.
        """
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

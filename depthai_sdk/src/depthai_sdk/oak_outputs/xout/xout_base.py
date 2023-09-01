from abc import ABC, abstractmethod
from typing import List, Optional, Callable

import depthai as dai

from depthai_sdk.classes.packets import FramePacket
from depthai_sdk.components.component import ComponentOutput


class StreamXout:
    def __init__(self, out: dai.Node.Output, name: Optional[str] = None):
        self.stream = out
        if name is not None:
            self.name = name
        else:
            node = out.getParent()
            self.name = f"{str(node.id)}_{out.name}"


class ReplayStream(StreamXout):
    def __init__(self, name: str):
        self.name = name


class XoutBase(ABC):
    def __init__(self) -> None:
        self._streams = [xout.name for xout in self.xstreams()]
        self._packet_name = None
        self._packet_name_postfix = None

        # It will get assigned later inside the BasePacketHandler class
        self.new_packet_callback: Callable = lambda x: None

    def get_packet_name(self) -> str:
        if self._packet_name is None:
            self._packet_name = ";".join([xout.name for xout in self.xstreams()])
        return self._packet_name + (f'_{self._packet_name_postfix}' if self._packet_name_postfix else '')

    def set_packet_name_postfix(self, postfix: str) -> None:
        """
        Set postfix to packet name.
        """
        self._packet_name_postfix = postfix

    def set_comp_out(self, comp_out: ComponentOutput) -> 'XoutBase':
        """
        Set ComponentOutput to Xout.
        """
        if comp_out.name is None:
            # If user hasn't specified component's output name, generate one
            comp_out.name = self.get_packet_name()
        else:
            # Otherwise, set packet name to user-specified one
            self._packet_name = comp_out.name
        return self

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
                # In case we have encoded frames, we need to set the codec
                if isinstance(p, FramePacket) and \
                        hasattr(self, 'get_codec') and \
                        self._fourcc is not None:
                    p.set_decode_codec(self.get_codec)

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
        fourcc = getattr(self, '_fourcc', None)
        return fourcc is not None and fourcc.lower() == 'hevc'

    def is_h264(self) -> bool:
        fourcc = getattr(self, '_fourcc', None)
        return fourcc is not None and fourcc.lower() == 'h264'

    def is_mjpeg(self) -> bool:
        fourcc = getattr(self, '_fourcc', None)
        return fourcc is not None and fourcc.lower() == 'mjpeg'

    def is_h26x(self) -> bool:
        return self.is_h264() or self.is_h265()

    def is_raw(self) -> bool:
        return type(self).__name__ == 'XoutFrames' and self._fourcc is None

    def is_depth(self) -> bool:
        return type(self).__name__ == 'XoutDepth'

    def is_disparity(self) -> bool:
        return type(self).__name__ == 'XoutDisparity'

    def is_imu(self):
        return type(self).__name__ == 'XoutIMU'

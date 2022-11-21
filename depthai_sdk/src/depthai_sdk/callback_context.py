from dataclasses import dataclass
from typing import Union, TypeVar, Dict

from depthai_sdk.classes.packets import IMUPacket, FramePacket
from depthai_sdk.recorders.video_writers import AbstractWriter
from depthai_sdk.visualize import Visualizer

P = TypeVar('P', bound=FramePacket)
R = TypeVar('R', bound=AbstractWriter)
Packet = Union[P, IMUPacket]


@dataclass
class CallbackContext:
    packet: Packet
    visualizer: Visualizer = None
    recorders: Dict[str, R] = None

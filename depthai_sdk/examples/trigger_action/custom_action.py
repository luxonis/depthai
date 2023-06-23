from pathlib import Path
from typing import Dict

import cv2

from depthai_sdk import OakCamera, FramePacket
from depthai_sdk.trigger_action import Action, DetectionTrigger


class MyAction(Action):
    """
    Saves the latest frame from the input stream to a file.
    """

    def __init__(self, inputs, dir_path):
        super().__init__(inputs)

        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)

        self.latest_packets = None

    def activate(self):
        print('+', self.latest_packets)
        if self.latest_packets:
            for stream_name, packet in self.latest_packets.items():
                print(f'Saving {stream_name} to {self.dir_path / f"{stream_name}.jpg"}')
                cv2.imwrite(str(self.dir_path / f'{stream_name}.jpg'), packet.frame)

    def on_new_packets(self, packets: Dict[str, FramePacket]) -> None:
        self.latest_packets = packets


with OakCamera() as oak:
    color = oak.create_camera('color', '1080p')
    nn = oak.create_nn('mobilenet-ssd', color)

    oak.trigger_action(
        trigger=DetectionTrigger(input=nn, min_detections={'person': 1}, cooldown=30),
        action=MyAction(inputs=[nn], dir_path='./images/')  # `action` can be Callable as well
    )

    oak.start(blocking=True)

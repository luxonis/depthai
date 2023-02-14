from typing import List, Optional

import depthai as dai

from depthai_sdk.oak_outputs.xout.xout_base import XoutBase


class OakDevice:
    def __init__(self):
        self.device: Optional[dai.Device] = None
        # fpsHandlers: Dict[str, FPS] = dict()
        self.oak_out_streams: List[XoutBase] = []

    @property
    def image_sensors(self) -> List[dai.CameraBoardSocket]:
        """
        Available imageSensors available on the camera
        """
        return self.device.getConnectedCameras()

    @property
    def info(self) -> dai.DeviceInfo:
        return self.device.getDeviceInfo()

    def init_callbacks(self, pipeline: dai.Pipeline):
        for node in pipeline.getAllNodes():
            if isinstance(node, dai.node.XLinkOut):
                stream_name = node.getStreamName()
                # self.fpsHandlers[name] = FPS()
                self.device.getOutputQueue(stream_name, maxSize=4, blocking=False).addCallback(
                    lambda name, msg: self.new_msg(name, msg)
                )

    def new_msg(self, name, msg):
        for sync in self.oak_out_streams:
            sync.new_msg(name, msg)

    def check_sync(self):
        """
        Checks whether there are new synced messages, non-blocking.
        """
        for sync in self.oak_out_streams:
            sync.check_queue(block=False)  # Don't block!

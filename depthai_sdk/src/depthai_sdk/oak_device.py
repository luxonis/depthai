import depthai as dai
from typing import Type, Dict, List
from .oak_outputs.xout_base import XoutBase


class OakDevice:
    device: dai.Device = None
    # fpsHandlers: Dict[str, FPS] = dict()
    oak_out_streams: List[XoutBase] = []

    @property
    def imageSensors(self) -> List[dai.CameraBoardSocket]:
        """
        Available imageSensors available on the camera
        """
        return self.device.getConnectedCameras()

    @property
    def info(self) -> dai.DeviceInfo:
        return self.device.getDeviceInfo()

    def initCallbacks(self, pipeline: dai.Pipeline):
        for node in pipeline.getAllNodes():
            if isinstance(node, dai.node.XLinkOut):
                name = node.getStreamName()
                # self.fpsHandlers[name] = FPS()
                self.device.getOutputQueue(name, maxSize=4, blocking=False).addCallback(
                    lambda name, msg: self.newMsg(name, msg))

    def newMsg(self, name, msg):
        for sync in self.oak_out_streams:
            sync.newMsg(name, msg)

        # if name not in self.fpsHandlers:
        #     self.fpsHandlers[name] = FPS()
        # self.fpsHandlers[name].next_iter()
        # print(name,' fps', self.fpsHandlers[name].fps())

    def checkSync(self):
        """
        Checks whether there are new synced messages, non-blocking.
        """
        for sync in self.oak_out_streams:
            sync.checkQueue(block=False)  # Don't block!

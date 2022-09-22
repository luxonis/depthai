import depthai as dai
from typing import Type, Dict, List
from .classes.xout_base import XoutBase
from .visualizing import FPS
from .replay import Replay


class OakDevice:
    device: dai.Device
    fpsHandlers: Dict[str, FPS] = dict()
    # str: Name (XLinkOut stream name, or replay stream)
    # Type: Component name, or Replay
    sync: List[XoutBase] = []

    @property
    def imageSensors(self) -> List[dai.CameraBoardSocket]:
        """
        Available imageSensors available on the camera
        """
        return self.device.getConnectedCameras()

    @property
    def info(self) -> dai.DeviceInfo:
        return self.device.getDeviceInfo()

    def initCallbacks(self):
        for xout in self.sync:
            for name in xout._streams:
                self.device.getOutputQueue(name, maxSize=4, blocking=False).addCallback(
                    lambda name, msg: self.newMsg(name, msg))
                self.fpsHandlers[name] = FPS()


    def newMsg(self, name, msg):
        print('New msg', name, msg)
        for sync in self.sync:
            sync.newMsg(name, msg)

        if name not in self.fpsHandlers:
            self.fpsHandlers[name] = FPS()
        self.fpsHandlers[name].next_iter()

    def checkSync(self):
        """
        Checks whether there are new synced messages, non-blocking.
        """
        for sync in self.sync:
            sync.checkQueue(block=False)  # Don't block!

import depthai as dai
from typing import Type, Dict, List
from .classes.xout_base import XoutBase, ReplayStream
from .visualizing import FPS
from .replay import Replay


class OakDevice:
    device: dai.Device
    fpsHandlers: Dict[str, FPS] = dict()
    # str: Name (XLinkOut stream name, or replay stream)
    # Type: Component name, or Replay
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

    def initCallbacks(self):
        for xout in self.oak_out_streams:
            for stream in xout.xstreams():
                if isinstance(stream, ReplayStream):
                    continue # Replay stream, we skip this to preserve bandwidth

                self.device.getOutputQueue(stream.name, maxSize=4, blocking=False).addCallback(
                    lambda name, msg: self.newMsg(stream.name, msg))
                self.fpsHandlers[stream.name] = FPS()


    def newMsg(self, name, msg):
        for sync in self.oak_out_streams:
            sync.newMsg(name, msg)

        if name not in self.fpsHandlers:
            self.fpsHandlers[name] = FPS()
        self.fpsHandlers[name].next_iter()

    def checkSync(self):
        """
        Checks whether there are new synced messages, non-blocking.
        """
        for sync in self.oak_out_streams:
            sync.checkQueue(block=False)  # Don't block!

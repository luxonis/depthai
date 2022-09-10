import depthai as dai
from typing import Type, Dict, List

from .syncing import BaseSync
from .visualizing import FPS
from .replay import Replay


class OakDevice:
    device: dai.Device

    fpsHandlers: Dict[str, FPS] = None

    # str: Name (XLinkOut stream name, or replay stream)
    # Type: Component name, or Replay
    queues: Dict[str, Type] = {}

    # Each Sync has its own Queue to which it pushes sycned msgs.
    # For each OAK cam you can have multiple sync mechanisms.
    sync: List[BaseSync] = []

    @property
    def imageSensors(self) -> List[dai.CameraBoardSocket]:
        """
        Available imageSensors available on the camera
        """
        return self.device.getConnectedCameras()

    @property
    def info(self) -> dai.DeviceInfo:
        return self.device.getDeviceInfo()

    _xoutNames: List[str] = None

    @property
    def xoutNames(self) -> List[str]:
        if not self._xoutNames:
            self._xoutNames = []
            for qName, qType in self.queues.items():
                if qType == Replay:
                    continue
                self._xoutNames.append(qName)
        return self._xoutNames

    _replayNames: List[str] = None

    @property
    def replayNames(self) -> List[str]:
        if not self._replayNames:
            self._replayNames = []
            for qName, qType in self.queues.items():
                if qType != Replay:
                    continue
                self._replayNames.append(qName)
        return self._replayNames

    def initCallbacks(self):
        for name in self.xoutNames:
            self.device.getOutputQueue(name, maxSize=4, blocking=False).addCallback(
                lambda name, msg: self.newMsg(name, msg))

    def enable_fps(self, enable: bool):
        """
        Enable or disable FPS measurement of each queue.
        @param enable: Whether to enable FPS measurement
        """
        self.fpsHandlers = dict() if enable else None

    def newMsg(self, name, msg):
        for sync in self.sync:
            sync.newMsg(name, msg)

        if self.fpsHandlers is not None:
            if name not in self.fpsHandlers:
                self.fpsHandlers[name] = FPS()
            self.fpsHandlers[name].next_iter()

    def checkSync(self):
        """
        Checks whether there are new synced messages, non-blocking.
        """
        for sync in self.sync:
            sync.checkQueue(block=False)  # Don't block!

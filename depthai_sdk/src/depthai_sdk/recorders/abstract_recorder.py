from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import depthai as dai


class Recorder(ABC):
    @abstractmethod
    def write(self, name: str, frame: dai.ImgFrame):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, path: Path, device: dai.Device, xouts: List['XoutFrames']):
        raise NotImplementedError()

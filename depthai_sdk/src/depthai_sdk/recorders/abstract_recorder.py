from abc import ABC, abstractmethod
import depthai as dai

class Recorder(ABC):
    @abstractmethod
    def write(self, name: str, frame: dai.ImgFrame):
        pass
    @abstractmethod
    def close(self):
        pass
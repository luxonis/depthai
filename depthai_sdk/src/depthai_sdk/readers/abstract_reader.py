from abc import ABC, abstractmethod
import array

class AbstractReader(ABC):
    @abstractmethod
    def read(self):
        pass
    @abstractmethod
    def getStreams(self) -> array:
        pass
    @abstractmethod
    def getShape(self, name: str) -> tuple:
        """
        Returns (width, height)
        """
        pass
    @abstractmethod
    def getStreams(self) -> array:
        pass
    @abstractmethod
    def close(self):
        pass
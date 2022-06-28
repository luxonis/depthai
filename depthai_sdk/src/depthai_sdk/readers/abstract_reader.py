from abc import ABC, abstractmethod

class AbstractReader(ABC):
    @abstractmethod
    def read(self):
        pass
    @abstractmethod
    # Returns (width, height)
    def getShape(self) -> tuple:
        pass
    @abstractmethod
    def close(self):
        pass
from abc import ABC, abstractmethod

class Recorder(ABC):
    @abstractmethod
    def write(self, name: str, frame):
        pass
    @abstractmethod
    def close(self):
        pass
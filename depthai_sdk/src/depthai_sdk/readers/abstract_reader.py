from abc import ABC, abstractmethod
import array

class AbstractReader(ABC):
    @abstractmethod
    def read(self):
        """
        Read a frame (or multiple frames) from the reader.
        @return: Single np.ndarray, or dict of frames and their names. None if frames weren't read or there was an error.
        """
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
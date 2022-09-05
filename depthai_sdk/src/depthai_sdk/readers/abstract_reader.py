from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractReader(ABC):
    @abstractmethod
    def read(self):
        """
        Read a frame (or multiple frames) from the reader.
        @return: Single np.ndarray, or dict of frames and their names. None if frames weren't read or there was an error.
        """
        pass

    @abstractmethod
    def getStreams(self) -> List[str]:
        pass
 
    @abstractmethod
    def getShape(self, name: str) -> Tuple[int, int]:
        """
        Returns (width, height)
        """
        pass

    @abstractmethod
    def close(self):
        pass
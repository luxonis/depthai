from abc import ABC, abstractmethod
from typing import List, Tuple

class AbstractReader(ABC):
    @abstractmethod
    def read(self):
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
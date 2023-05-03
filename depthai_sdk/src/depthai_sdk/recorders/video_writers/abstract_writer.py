from abc import ABC


class AbstractWriter(ABC):
    def write(self, frame):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

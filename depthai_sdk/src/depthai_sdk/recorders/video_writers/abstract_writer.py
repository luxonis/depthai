from abc import ABC


class AbstractWriter(ABC):
    def write(self, frame):
        raise NotImplementedError()

    def write_from_buffer(self, bufname: str, n_elems: int):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

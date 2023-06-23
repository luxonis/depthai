import time


class FPS:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self) -> float:
        diff = self.timestamp - self.start
        return self.frame_cnt / diff if diff != 0 else 0.0

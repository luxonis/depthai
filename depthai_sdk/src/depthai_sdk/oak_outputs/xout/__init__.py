try:
    import cv2
except ImportError:
    cv2 = None


class Clickable:
    def __init__(self, decay_step: int = 30):
        super().__init__()
        self.buffer = None
        self.decay_step = decay_step

    def on_click_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.buffer = ([0, param[0][y, x], [x, y]])

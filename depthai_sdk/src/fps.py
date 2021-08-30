import time
import cv2


class FPSHandler:
    fps_bg_color = (0, 0, 0)
    fps_color = (255, 255, 255)
    fps_type = cv2.FONT_HERSHEY_SIMPLEX
    fps_line_type = cv2.LINE_AA

    def __init__(self, cap=None):
        self.timestamp = time.monotonic()
        self.start = None
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self.useCamera = cap is None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if self.start is None:
            self.start = time.monotonic()

        if not self.useCamera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.monotonic()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.monotonic()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            time_diff = time.monotonic() - self.ticks[name]
            return self.ticks_cnt[name] / time_diff if time_diff != 0 else 0
        else:
            return 0

    def fps(self):
        if self.start is None:
            return 0
        time_diff = self.timestamp - self.start
        return self.frame_cnt / time_diff if time_diff != 0 else 0

    def print_status(self):
        print("=== TOTAL FPS ===")
        for name in self.ticks:
            print(f"[{name}]: {self.tick_fps(name):.1f}")

    def draw_fps(self, frame, name):
        frame_fps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, frame_fps, (5, 15), self.fps_type, 0.5, self.fps_bg_color, 4, self.fps_line_type)
        cv2.putText(frame, frame_fps, (5, 15), self.fps_type, 0.5, self.fps_color, 1, self.fps_line_type)

        if "nn" in self.ticks:
            cv2.putText(frame, f"NN FPS:  {round(self.tick_fps('nn'), 1)}", (5, 30), self.fps_type, 0.5, self.fps_bg_color, 4, self.fps_line_type)
            cv2.putText(frame, f"NN FPS:  {round(self.tick_fps('nn'), 1)}", (5, 30), self.fps_type, 0.5, self.fps_color, 1, self.fps_line_type)

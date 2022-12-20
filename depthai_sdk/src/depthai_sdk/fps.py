import collections
import time

import cv2


class FPSHandler:
    """
    Class that handles all FPS-related operations. Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on it's FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    _fpsBgColor = (0, 0, 0)
    _fpsColor = (255, 255, 255)
    _fpsType = cv2.FONT_HERSHEY_SIMPLEX
    _fpsLineType = cv2.LINE_AA

    def __init__(self, cap=None, maxTicks=100):
        """
        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            maxTicks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if maxTicks < 2:
            raise ValueError(f"Proviced maxTicks value must be 2 or higher (supplied: {maxTicks})")

        self._maxTicks = maxTicks

    def nextIter(self):
        """
        Marks the next iteration of the processing loop. Will use :obj:`time.sleep` method if initialized with video file
        object
        """
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frameDelay = 1.0 / self._framerate
            delay = (self._timestamp + frameDelay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._timestamp = time.monotonic()
        self._iterCnt += 1

    def tick(self, name):
        """
        Marks a point in time for specified name

        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tickFps(self, name):
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            timeDiff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / timeDiff if timeDiff != 0 else 0.0
        else:
            return 0.0

    def fps(self):
        """
        Calculates FPS value based on :func:`nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        timeDiff = self._timestamp - self._start
        return self._iterCnt / timeDiff if timeDiff != 0 else 0.0

    def printStatus(self):
        """
        Prints total FPS for all names stored in :func:`tick` calls
        """
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tickFps(name):.1f}")

    def drawFps(self, frame, name):
        """
        Draws FPS values on requested frame, calculated based on specified name

        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frameFps = f"{name.upper()} FPS: {round(self.tickFps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, frameFps, (5, 15), self._fpsType, 0.5, self._fpsBgColor, 4, self._fpsLineType)
        cv2.putText(frame, frameFps, (5, 15), self._fpsType, 0.5, self._fpsColor, 1, self._fpsLineType)

        if "nn" in self._ticks:
            cv2.putText(frame, f"NN FPS:  {round(self.tickFps('nn'), 1)}", (5, 30), self._fpsType, 0.5,
                        self._fpsBgColor, 4, self._fpsLineType)
            cv2.putText(frame, f"NN FPS:  {round(self.tickFps('nn'), 1)}", (5, 30), self._fpsType, 0.5, self._fpsColor,
                        1, self._fpsLineType)

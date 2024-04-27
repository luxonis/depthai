import depthai as dai
from itertools import cycle
import logging

logger = logging.getLogger(__name__)

LIMITS = {
    "confidence_threshold": (0, 255),
    "bilateral_sigma": (0, 255),
    "range": (0, 65535),
    "lrc_threshold": (0, 10),
    "dot_projector": (0, 1200),
    "illumination_led": (0, 1500),
}


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class ToFControl:
    def __init__(self, device: dai.Device):
        self.queue = None
        ctrl = dai.StereoDepthConfig()
        self.raw_cfg = ctrl.get()

    def set_input_queue(self, queue: dai.DataInputQueue):
        self.queue = queue

    def send_controls(self, tof_control: dai.RawToFConfig):
        """
        Send controls to the ToF node.
        """
        if self.queue is None:
            logger.error("Cannot send controls when replaying.")
            return

        logger.info(f"Sending controls to ToF node: {tof_control}")
        self.queue.send(tof_control)

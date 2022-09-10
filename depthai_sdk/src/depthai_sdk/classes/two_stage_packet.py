from typing import Optional, List, Tuple
import depthai as dai

class TwoStageSyncPacket:
    """
    Packet of (two-stage NN) synced messages
    """
    # global doesn't work as expected here..
    _labels: Optional[List[int]] = None
    _scaleBbs: Optional[Tuple[int, int]] = None

    def __init__(self, labels, scaleBbs):
        self._labels = labels
        self._scaleBbs = scaleBbs

    frame: dai.ImgFrame = None
    _dets: dai.ImgDetections = None
    recognitions: List[dai.NNData] = []

    @property
    def dets(self) -> dai.ImgDetections:
        return self._dets

    @dets.setter
    def dets(self, dets: dai.ImgDetections):
        # Used to match the scaled bounding boxes by the 2-stage NN script node
        self._dets = dets
        if self._scaleBbs is None: return  # No scaling required, ignore

        for det in self._dets.detections:
            # Skip resizing BBs if we have whitelist and the detection label is not on it
            if self._labels and det.label not in self._labels: continue
            det.xmin -= self._scaleBbs[0] / 100
            det.ymin -= self._scaleBbs[1] / 100
            det.xmax += self._scaleBbs[0] / 100
            det.ymax += self._scaleBbs[1] / 100

    def synced(self) -> bool:
        """
        Messages are in sync if:
            - dets is not None
            - We have at least one ImgFrame
            - number of recognition msgs is sufficient
        """
        return (self.dets and self.frame and len(self.recognitions) == self._required_recognitions())

    def _required_recognitions(self) -> int:
        """
        Required recognition results for this packet, which depends on number of detections (and white-list labels)
        """
        if self._labels:
            return len([det for det in self.dets.detections if det.label in self._labels])
        else:
            return len(self.dets.detections)
import os
from dataclasses import replace
from enum import Enum
from typing import List, Tuple, Optional

import cv2
import numpy as np
from depthai import ImgDetection

from .configs import VisConfig
from .objects import VisDetections, VisObject, VisText
from ..oak_outputs.normalize_bb import NormalizeBoundingBox


class Platform(Enum):
    ROBOTHUB = 'robothub'
    PC = 'pc'


overlay_priority = {
    VisDetections: 0,
    VisText: 1
}


class NewVisualizer:
    # Constants
    IS_INTERACTIVE = 'DISPLAY' in os.environ or os.name == 'nt'

    # Fields
    objects: List[VisObject]

    def __init__(self):
        self.platform: Platform = self._detect_platform()
        self.objects: List[VisObject] = []

        self.config = VisConfig()

    def _detect_platform(self) -> Platform:
        return Platform.ROBOTHUB if self.IS_INTERACTIVE else Platform.PC

    def add_object(self, obj: VisObject) -> 'NewVisualizer':
        self.objects.append(obj)
        return self

    def add_detections(self,
                       detections: List[ImgDetection],
                       normalizer: NormalizeBoundingBox,
                       label_map: List[Tuple[str, Tuple]] = None
                       ) -> 'NewVisualizer':
        detection_overlay = VisDetections(detections, normalizer, label_map)
        self.add_object(detection_overlay)
        return self

    def add_text(self, text: str, coords: Tuple[int, int]) -> 'NewVisualizer':
        text_overlay = VisText(text, coords)
        self.add_object(text_overlay)
        return self

    def draw(self, frame: np.ndarray, name: Optional[str] = 'Frames') -> None:
        if self.IS_INTERACTIVE:
            for obj in self.objects:
                obj.set_config(self.config).draw(frame)

            cv2.imshow(name, frame)

            self.objects.clear()  # Clear objects after drawing
        else:
            pass  # TODO encode/serialize and send everything to robothub

    def configure_bbox(self, **kwargs: dict) -> 'NewVisualizer':
        self.config.detection = replace(self.config.detection, **kwargs)
        return self

    def configure_text(self, **kwargs: dict) -> 'NewVisualizer':
        self.config.text = replace(self.config.text, **kwargs)
        return self

    def sort_objects(self) -> None:
        """Sort overlays by priority (in-place)."""
        overlays = sorted(self.objects, key=lambda overlay: overlay_priority[type(overlay)])
        self.objects = overlays

    def serialize(self) -> dict:
        # TODO serialization
        pass

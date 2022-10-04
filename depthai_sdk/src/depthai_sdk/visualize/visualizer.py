import os
from enum import Enum
from typing import List, Tuple, Optional

import cv2
import numpy as np
from depthai import ImgDetection

from .configs import NewVisConfig
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

        self.config = NewVisConfig()

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
        detection_overlay = VisDetections(detections, self.config.detection_config, normalizer, label_map)
        self.add_object(detection_overlay)
        return self

    def add_text(self,
                 text: str,
                 coords: Tuple[int, int],
                 scale: float = 1.0,
                 bg_color: Tuple[int, int, int] = None,
                 color: Tuple[int, int, int] = None
                 ) -> 'NewVisualizer':
        text_overlay = VisText(text, coords, scale, bg_color, color)
        self.add_object(text_overlay)
        return self

    def draw(self, frame: np.ndarray, name: Optional[str] = 'Frames') -> None:
        if self.IS_INTERACTIVE:
            for obj in self.objects:
                obj.draw(frame)

            cv2.imshow(name, frame)

            self.objects.clear()  # Clear objects after drawing
        else:
            pass  # TODO encode/serialize and send everything to robothub

    def configure_bbox(self,
                       thickness: int = None,
                       fill_transparency: float = None,
                       box_roundness: float = None,
                       color: Tuple[int, int, int] = None
                       ) -> 'NewVisualizer':
        """
        Configure bounding box.

        Args:
            thickness: Thickness of the bounding box.
            fill_transparency: Transparency of the bounding box.
            box_roundness: Roundness of the bounding box.
            color: Color of the bounding box.

        Returns:
            NewVisualizer (self) instance.
        """
        config = self.config.detection_config
        config.thickness = thickness or config.thickness
        config.fill_transparency = fill_transparency or config.fill_transparency
        config.box_roundness = box_roundness or config.box_roundness
        config.color = color or config.color
        return self

    def sort_objects(self) -> None:
        """Sort overlays by priority (in-place)."""
        overlays = sorted(self.objects, key=lambda overlay: overlay_priority[type(overlay)])
        self.objects = overlays

    def serialize(self) -> dict:
        # TODO serialization
        pass

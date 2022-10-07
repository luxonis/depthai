import os
from dataclasses import replace
from enum import Enum
from typing import List, Tuple, Optional, Union

import cv2
import depthai as dai
import numpy as np
from depthai import ImgDetection

from .configs import VisConfig, TextPosition
from .objects import VisDetections, VisObject, VisText, VisTrail
from ..classes.packets import _TrackingDetection
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
                       frame_shape: Tuple[int, ...],
                       detections: List[Union[ImgDetection, dai.Tracklet]],
                       normalizer: NormalizeBoundingBox,
                       label_map: List[Tuple[str, Tuple]] = None,
                       spatial_points: List[dai.Point3f] = None,
                       is_spatial=False) -> 'NewVisualizer':
        detection_overlay = VisDetections(frame_shape, detections, normalizer,
                                          label_map, spatial_points, is_spatial)
        self.add_object(detection_overlay)
        return self

    def add_text(self,
                 text: str,
                 coords: Tuple[int, int] = None,
                 bbox: Tuple[int, int, int, int] = None,
                 position: TextPosition = TextPosition.TOP_LEFT,
                 padding: int = 10) -> 'NewVisualizer':
        text_overlay = VisText(text, coords, bbox, position, padding)
        self.add_object(text_overlay)
        return self

    def add_trail(self,
                  detections: List[List[_TrackingDetection]],
                  tracklets: List[List[dai.Tracklet]]) -> 'NewVisualizer':
        trail = VisTrail(detections, tracklets)
        self.add_object(trail)
        return self

    def draw(self, frame: np.ndarray, name: Optional[str] = 'Frames') -> None:
        if self.IS_INTERACTIVE:
            for obj in self.objects:
                obj.set_config(self.config).draw(frame)

            img_scale = self.config.img_scale
            if img_scale:
                if isinstance(img_scale, Tuple):
                    frame = cv2.resize(frame, img_scale)  # Resize frame
                elif isinstance(img_scale, float) and img_scale != 1.0:
                    frame = cv2.resize(frame, (
                        int(frame.shape[1] * img_scale),
                        int(frame.shape[0] * img_scale)
                    ))

            cv2.imshow(name, frame)

            self.objects.clear()  # Clear objects after drawing
        else:
            pass  # TODO encode/serialize and send everything to robothub

    def configure_output(self, **kwargs: dict) -> 'NewVisualizer':
        self.config = replace(self.config, **kwargs)
        return self

    def configure_bbox(self, **kwargs: dict) -> 'NewVisualizer':
        self.config.detection = replace(self.config.detection, **kwargs)
        return self

    def configure_text(self, **kwargs: dict) -> 'NewVisualizer':
        self.config.text = replace(self.config.text, **kwargs)
        return self

    def configure_tracking(self, **kwargs: dict) -> 'NewVisualizer':
        self.config.tracking = replace(self.config.tracking, **kwargs)
        return self

    def sort_objects(self) -> None:
        """Sort overlays by priority (in-place)."""
        overlays = sorted(self.objects, key=lambda overlay: overlay_priority[type(overlay)])
        self.objects = overlays

    def serialize(self) -> dict:
        # TODO serialization
        pass

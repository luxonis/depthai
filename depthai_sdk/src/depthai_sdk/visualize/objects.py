from abc import ABC, abstractmethod
from typing import Tuple, List

import cv2
from depthai import ImgDetection

from .configs import DetectionConfig
from ..oak_outputs.normalize_bb import NormalizeBoundingBox
from ..oak_outputs.visualizer_helper import rectangle


class VisObject(ABC):
    @abstractmethod
    def draw(self, frame):
        pass


class VisDetections(VisObject):
    def __init__(self,
                 detections: List[ImgDetection],
                 detection_config: DetectionConfig,
                 normalizer: NormalizeBoundingBox,
                 label_map: List[Tuple[str, Tuple]] = None):
        super().__init__()
        self.detections = detections
        self.detection_config = detection_config
        self.normalizer = normalizer
        self.label_map = label_map

    def draw(self, frame):
        for detection in self.detections:
            # Get normalized bounding box
            bbox = detection.xmin, detection.ymin, detection.xmax, detection.ymax
            normalized_bbox = self.normalizer.normalize(frame, bbox)

            if self.label_map:
                label, color = self.label_map[detection.label]
            else:
                label = str(detection.label)
                color = self.detection_config.color

            # Place label in the bounding box (top left corner)
            VisText(text=label, coords=(normalized_bbox[0] + 5, normalized_bbox[1] + 25), scale=0.9).draw(frame)

            # Draw bounding box
            rectangle(src=frame,
                      bbox=normalized_bbox,
                      color=color,
                      thickness=self.detection_config.thickness,
                      radius=self.detection_config.box_roundness,
                      alpha=self.detection_config.fill_transparency)


class VisText(VisObject):
    # Constants
    DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    DEFAULT_LINE_TYPE = cv2.LINE_AA
    DEFAULT_TEXT_COLOR = (255, 255, 255)
    DEFAULT_BG_COLOR = (0, 0, 0)

    def __init__(self,
                 text: str,
                 coords: Tuple[int, int],
                 scale: float = 1.0,
                 bg_color: Tuple[int, int, int] = None,
                 color: Tuple[int, int, int] = None
                 ):
        super().__init__()
        self.text = text
        self.coords = coords
        self.scale = scale
        self.bg_color = bg_color
        self.color = color

    def draw(self, frame):
        """
        Draw text on the frame.

        Args:
            frame: np.ndarray

        Returns:
            None
        """
        # Background
        cv2.putText(img=frame,
                    text=self.text,
                    org=self.coords,
                    fontFace=self.DEFAULT_FONT,
                    fontScale=self.scale,
                    color=self.bg_color or self.DEFAULT_BG_COLOR,
                    thickness=int(self.scale * 3),
                    lineType=self.DEFAULT_LINE_TYPE)

        # Front text
        cv2.putText(img=frame,
                    text=self.text,
                    org=self.coords,
                    fontFace=self.DEFAULT_FONT,
                    fontScale=self.scale,
                    color=tuple(map(int, self.color or self.DEFAULT_TEXT_COLOR)),
                    thickness=int(self.scale),
                    lineType=self.DEFAULT_LINE_TYPE)


class VisPolygon(VisObject):
    def __init__(self, polygon):
        super().__init__()
        self.polygon = polygon

    def draw(self, frame):
        pass

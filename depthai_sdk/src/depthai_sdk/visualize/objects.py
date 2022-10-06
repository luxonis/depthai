from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import cv2
import numpy as np
from depthai import ImgDetection

from .configs import VisConfig, BboxStyle, TextPosition
from ..oak_outputs.normalize_bb import NormalizeBoundingBox


class VisObject(ABC):
    def __init__(self):
        self.config = None

    def set_config(self, config: VisConfig) -> "VisObject":
        self.config = config
        return self

    @abstractmethod
    def draw(self, frame) -> None:
        pass

    def draw_bbox(self,
                  img: np.ndarray,
                  pt1: Tuple[int, int],
                  pt2: Tuple[int, int],
                  color: Tuple[int, int, int],
                  thickness: int,
                  r: int,
                  line_width: int,
                  line_height: int) -> None:
        """
        Draw a rounded rectangle on the image (in-place).

        Args:
            img: Image to draw on.
            pt1: Top-left corner of the rectangle.
            pt2: Bottom-right corner of the rectangle.
            color: Rectangle color.
            thickness: Rectangle line thickness.
            r: Radius of the rounded corners.
            line_width: Width of the rectangle line.
            line_height: Height of the rectangle line.

        Returns:
            None
        """
        x1, y1 = pt1
        x2, y2 = pt2

        if line_width == 0:
            line_width = np.abs(x2 - x1)
            line_width -= 2 * r if r > 0 else 0  # Adjust for rounded corners

        if line_height == 0:
            line_height = np.abs(y2 - y1)
            line_height -= 2 * r if r > 0 else 0  # Adjust for rounded corners

        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + line_width, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + line_height), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - line_width, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + line_height), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + line_width, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - line_height), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - line_width, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - line_height), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        # Fill the area
        alpha = self.config.detection.fill_transparency
        if alpha > 0:
            overlay = img.copy()

            thickness = -1
            bbox = (pt1[0], pt1[1], pt2[0], pt2[1])

            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[2], bbox[3])
            top_right = (bottom_right[0], top_left[1])
            bottom_left = (top_left[0], bottom_right[1])

            top_left_main_rect = (int(top_left[0] + r), int(top_left[1]))
            bottom_right_main_rect = (int(bottom_right[0] - r), int(bottom_right[1]))

            top_left_rect_left = (top_left[0], top_left[1] + r)
            bottom_right_rect_left = (bottom_left[0] + r, bottom_left[1] - r)

            top_left_rect_right = (top_right[0] - r, top_right[1] + r)
            bottom_right_rect_right = (bottom_right[0], bottom_right[1] - r)

            all_rects = [
                [top_left_main_rect, bottom_right_main_rect],
                [top_left_rect_left, bottom_right_rect_left],
                [top_left_rect_right, bottom_right_rect_right]
            ]

            [cv2.rectangle(overlay, pt1=rect[0], pt2=rect[1], color=color, thickness=thickness) for rect in all_rects]

            cv2.ellipse(overlay, (top_left[0] + r, top_left[1] + r), (r, r), 180.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (top_right[0] - r, top_right[1] + r), (r, r), 270.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (bottom_right[0] - r, bottom_right[1] - r), (r, r), 0.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (bottom_left[0] + r, bottom_left[1] - r), (r, r), 90.0, 0, 90, color, thickness)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class VisDetections(VisObject):
    def __init__(self,
                 detections: List[ImgDetection],
                 normalizer: NormalizeBoundingBox,
                 label_map: List[Tuple[str, Tuple]] = None):
        super().__init__()
        self.detections = detections
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
                color = self.config.detection.color

            # Draw bounding box
            self.draw_stylized_bbox(
                img=frame,
                pt1=(normalized_bbox[0], normalized_bbox[1]),
                pt2=(normalized_bbox[2], normalized_bbox[3]),
                color=color,
                thickness=self.config.detection.thickness
            )

            if not self.config.detection.hide_label:
                # Place label in the bounding box
                VisText(text=label, bbox=normalized_bbox, position=self.config.detection.label_position) \
                    .set_config(self.config) \
                    .draw(frame)

    def draw_stylized_bbox(self,
                           img: np.ndarray,
                           pt1: Tuple[int, int],
                           pt2: Tuple[int, int],
                           color: Tuple[int, int, int],
                           thickness: int) -> None:
        box_w = pt2[0] - pt1[0]
        box_h = pt2[1] - pt1[1]
        line_width = int(box_w * self.config.detection.line_width) // 2
        line_height = int(box_h * self.config.detection.line_height) // 2
        roundness = int(self.config.detection.box_roundness)

        if self.config.detection.bbox_style == BboxStyle.RECTANGLE:
            self.draw_bbox(img, pt1, pt2, color, thickness, 0, line_width=0, line_height=0)
        elif self.config.detection.bbox_style == BboxStyle.CORNERS:
            self.draw_bbox(img, pt1, pt2, color, thickness, 0, line_width=line_width, line_height=line_height)
        elif self.config.detection.bbox_style == BboxStyle.ROUNDED_RECTANGLE:
            self.draw_bbox(img, pt1, pt2, color, thickness, roundness, line_width=0, line_height=0)
        elif self.config.detection.bbox_style == BboxStyle.ROUNDED_CORNERS:
            self.draw_bbox(img, pt1, pt2, color, thickness, roundness, line_width=line_width, line_height=line_height)


class VisText(VisObject):
    def __init__(self,
                 text: str,
                 coords: Tuple[int, int] = None,
                 bbox: Union[np.ndarray, Tuple[int, int, int, int]] = None,
                 position: TextPosition = TextPosition.TOP_LEFT,
                 padding: int = 10
                 ):
        super().__init__()
        self.text = text
        self.coords = coords
        self.bbox = bbox
        self.position = position
        self.padding = padding

    def draw(self, frame: np.ndarray) -> None:
        """
        Draw text on the frame.

        Args:
            frame: np.ndarray

        Returns:
            None
        """
        text_config = self.config.text

        self.coords = self.coords or self.get_relative_position(frame=frame,
                                                                bbox=self.bbox,
                                                                position=self.position,
                                                                padding=self.padding)

        # Background
        cv2.putText(img=frame,
                    text=self.text,
                    org=self.coords,
                    fontFace=text_config.font_face,
                    fontScale=text_config.font_scale,
                    color=text_config.bg_color,
                    thickness=int(text_config.font_scale * 3),
                    lineType=text_config.line_type)

        # Front text
        cv2.putText(img=frame,
                    text=self.text,
                    org=self.coords,
                    fontFace=text_config.font_face,
                    fontScale=text_config.font_scale,
                    color=text_config.font_color,
                    thickness=int(text_config.font_thickness),
                    lineType=text_config.line_type)

    def get_relative_position(self,
                              frame: np.ndarray,
                              bbox: Union[np.ndarray, Tuple[int, int, int, int]],
                              position: TextPosition,
                              padding: int):
        """
        Get relative position of the text.
        """
        frame_h, frame_w = frame.shape[0], frame.shape[1]

        if bbox is None:
            bbox = (0, 0, frame_w, frame_h)

        text_config = self.config.text
        text_size = cv2.getTextSize(text=self.text,
                                    fontFace=text_config.font_face,
                                    fontScale=text_config.font_scale,
                                    thickness=text_config.font_thickness)[0]

        x, y = bbox[0], bbox[1]

        y_pos = position.value % 10
        if y_pos == 0:  # Y top
            y = bbox[1] + text_size[1] + padding
        elif y_pos == 1:  # Y mid
            y = (bbox[1] + bbox[3]) // 2 + text_size[1] // 2
        elif y_pos == 2:  # Y bottom
            y = bbox[3] - padding

        x_pos = position.value // 10
        if x_pos == 0:  # X Left
            x = bbox[0] + padding
        elif x_pos == 1:  # X mid
            x = (bbox[0] + bbox[2]) // 2 - text_size[0] // 2
        elif x_pos == 2:  # X right
            x = bbox[2] - text_size[0] - padding

        return x, y


class VisPolygon(VisObject):
    def __init__(self, polygon):
        super().__init__()
        self.polygon = polygon

    def draw(self, frame):
        pass

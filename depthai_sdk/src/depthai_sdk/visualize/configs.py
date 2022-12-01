from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple

import cv2


class TextPosition(IntEnum):
    """Where on frame do we want to print text."""
    TOP_LEFT = 0
    MID_LEFT = 1
    BOTTOM_LEFT = 2

    TOP_MID = 10
    MID = 11
    BOTTOM_MID = 12

    TOP_RIGHT = 20
    MID_RIGHT = 21
    BOTTOM_RIGHT = 22


class BboxStyle(IntEnum):
    """How do we want to draw bounding box."""
    RECTANGLE = 0
    CORNERS = 1

    ROUNDED_RECTANGLE = 10
    ROUNDED_CORNERS = 11


class StereoColor:
    GRAY = 0
    RGB = 1
    RGBD = 2


@dataclass
class OutputConfig:
    """Configuration for output of the visualizer."""
    img_scale: float = 1.0
    show_fps: bool = False
    clickable: bool = True


@dataclass
class StereoConfig:
    colorize: StereoColor = StereoColor.RGB
    colormap: int = cv2.COLORMAP_JET
    wls_filter: bool = False
    wls_lambda: float = 1500
    wls_sigma: float = 1.5


@dataclass
class DetectionConfig:
    """Configuration for drawing bounding boxes."""
    thickness: int = 1
    fill_transparency: float = 0.15
    box_roundness: int = 0
    color: Tuple[int, int, int] = None
    bbox_style: BboxStyle = BboxStyle.RECTANGLE
    line_width: float = 0.5
    line_height: float = 0.5
    hide_label: bool = False
    label_position: TextPosition = TextPosition.TOP_LEFT
    label_padding: int = 10


@dataclass
class TextConfig:
    """Configuration for drawing labels."""
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_color: Tuple[int, int, int] = (255, 255, 255)
    font_transparency: float = 0.5
    font_scale: float = 1.0
    font_thickness: int = 2
    font_position: TextPosition = TextPosition.TOP_LEFT

    bg_transparency: float = 0.5
    bg_color: Tuple[int, int, int] = (0, 0, 0)

    line_type: int = cv2.LINE_AA

    auto_scale: bool = True


@dataclass
class TrackingConfig:
    """Configuration for drawing tracking bounding boxes."""
    max_length: int = -1
    deletion_lost_threshold: int = 5
    line_thickness: int = 1
    fading_tails: bool = False
    line_color: Tuple[int, int, int] = (255, 255, 255)
    line_type: int = cv2.LINE_AA
    bg_color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class CircleConfig:
    """Configuration for drawing circles."""
    thickness: int = 1
    color: Tuple[int, int, int] = (255, 255, 255)
    line_type: int = cv2.LINE_AA


@dataclass
class VisConfig:
    """Configuration for visualizer."""

    output: OutputConfig = field(default_factory=OutputConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    circle: CircleConfig = field(default_factory=CircleConfig)

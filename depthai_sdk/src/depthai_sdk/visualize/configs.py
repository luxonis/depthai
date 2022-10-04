from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple


class LabelPosition(IntEnum):
    """
    Where on frame do we want to print text.
    """
    TopLeft = 0
    MidLeft = 1
    BottomLeft = 2
    TopMid = 10
    Mid = 11
    BottomMid = 12
    TopRight = 20
    MidRight = 21
    BottomRight = 22


@dataclass
class DetectionConfig:
    thickness: int = -1
    fill_transparency: float = 0.15
    box_roundness: float = 0.0
    color: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class TextConfig:
    font_scale: float = 1.0


@dataclass
class LabelConfig:
    transparency: float = 0.5
    font_scale: float = 1.0
    thickness: int = 2
    position: LabelPosition = LabelPosition.TopLeft
    bg_transparency: float = 0.5
    color: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class NewVisConfig:
    img_scale: float = 1.0
    show_fps: bool = False

    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    label_config: LabelConfig = field(default_factory=LabelConfig)

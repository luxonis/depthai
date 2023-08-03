from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


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


class StereoColor(IntEnum):
    GRAY = 1
    RGB = 2
    RGBD = 3


@dataclass
class OutputConfig:
    """Configuration for output of the visualizer."""
    img_scale: float = 1.0
    show_fps: bool = False
    clickable: bool = True


@dataclass
class StereoConfig:
    colorize: StereoColor = StereoColor.RGB
    # cv2.COLORMAP_JET. This was hardcoded, as we want to have an array, because we later invert it / invalidate values 0
    colormap: np.ndarray = field(default_factory=lambda: np.array([[[128,0,0]],[[132,0,0]],[[136,0,0]],[[140,0,0]],[[144,0,0]],[[148,0,0]],[[152,0,0]],[[156,0,0]],[[160,0,0]],[[164,0,0]],[[168,0,0]],[[172,0,0]],[[176,0,0]],[[180,0,0]],[[184,0,0]],[[188,0,0]],[[192,0,0]],[[196,0,0]],[[200,0,0]],[[204,0,0]],[[208,0,0]],[[212,0,0]],[[216,0,0]],[[220,0,0]],[[224,0,0]],[[228,0,0]],[[232,0,0]],[[236,0,0]],[[240,0,0]],[[244,0,0]],[[248,0,0]],[[252,0,0]],[[255,0,0]],[[255,4,0]],[[255,8,0]],[[255,12,0]],[[255,16,0]],[[255,20,0]],[[255,24,0]],[[255,28,0]],[[255,32,0]],[[255,36,0]],[[255,40,0]],[[255,44,0]],[[255,48,0]],[[255,52,0]],[[255,56,0]],[[255,60,0]],[[255,64,0]],[[255,68,0]],[[255,72,0]],[[255,76,0]],[[255,80,0]],[[255,84,0]],[[255,88,0]],[[255,92,0]],[[255,96,0]],[[255,100,0]],[[255,104,0]],[[255,108,0]],[[255,112,0]],[[255,116,0]],[[255,120,0]],[[255,124,0]],[[255,128,0]],[[255,132,0]],[[255,136,0]],[[255,140,0]],[[255,144,0]],[[255,148,0]],[[255,152,0]],[[255,156,0]],[[255,160,0]],[[255,164,0]],[[255,168,0]],[[255,172,0]],[[255,176,0]],[[255,180,0]],[[255,184,0]],[[255,188,0]],[[255,192,0]],[[255,196,0]],[[255,200,0]],[[255,204,0]],[[255,208,0]],[[255,212,0]],[[255,216,0]],[[255,220,0]],[[255,224,0]],[[255,228,0]],[[255,232,0]],[[255,236,0]],[[255,240,0]],[[255,244,0]],[[255,248,0]],[[255,252,0]],[[254,255,2]],[[250,255,6]],[[246,255,10]],[[242,255,14]],[[238,255,18]],[[234,255,22]],[[230,255,26]],[[226,255,30]],[[222,255,34]],[[218,255,38]],[[214,255,42]],[[210,255,46]],[[206,255,50]],[[202,255,54]],[[198,255,58]],[[194,255,62]],[[190,255,66]],[[186,255,70]],[[182,255,74]],[[178,255,78]],[[174,255,82]],[[170,255,86]],[[166,255,90]],[[162,255,94]],[[158,255,98]],[[154,255,102]],[[150,255,106]],[[146,255,110]],[[142,255,114]],[[138,255,118]],[[134,255,122]],[[130,255,126]],[[126,255,130]],[[122,255,134]],[[118,255,138]],[[114,255,142]],[[110,255,146]],[[106,255,150]],[[102,255,154]],[[98,255,158]],[[94,255,162]],[[90,255,166]],[[86,255,170]],[[82,255,174]],[[78,255,178]],[[74,255,182]],[[70,255,186]],[[66,255,190]],[[62,255,194]],[[58,255,198]],[[54,255,202]],[[50,255,206]],[[46,255,210]],[[42,255,214]],[[38,255,218]],[[34,255,222]],[[30,255,226]],[[26,255,230]],[[22,255,234]],[[18,255,238]],[[14,255,242]],[[10,255,246]],[[6,255,250]],[[1,255,254]],[[0,252,255]],[[0,248,255]],[[0,244,255]],[[0,240,255]],[[0,236,255]],[[0,232,255]],[[0,228,255]],[[0,224,255]],[[0,220,255]],[[0,216,255]],[[0,212,255]],[[0,208,255]],[[0,204,255]],[[0,200,255]],[[0,196,255]],[[0,192,255]],[[0,188,255]],[[0,184,255]],[[0,180,255]],[[0,176,255]],[[0,172,255]],[[0,168,255]],[[0,164,255]],[[0,160,255]],[[0,156,255]],[[0,152,255]],[[0,148,255]],[[0,144,255]],[[0,140,255]],[[0,136,255]],[[0,132,255]],[[0,128,255]],[[0,124,255]],[[0,120,255]],[[0,116,255]],[[0,112,255]],[[0,108,255]],[[0,104,255]],[[0,100,255]],[[0,96,255]],[[0,92,255]],[[0,88,255]],[[0,84,255]],[[0,80,255]],[[0,76,255]],[[0,72,255]],[[0,68,255]],[[0,64,255]],[[0,60,255]],[[0,56,255]],[[0,52,255]],[[0,48,255]],[[0,44,255]],[[0,40,255]],[[0,36,255]],[[0,32,255]],[[0,28,255]],[[0,24,255]],[[0,20,255]],[[0,16,255]],[[0,12,255]],[[0,8,255]],[[0,4,255]],[[0,0,255]],[[0,0,252]],[[0,0,248]],[[0,0,244]],[[0,0,240]],[[0,0,236]],[[0,0,232]],[[0,0,228]],[[0,0,224]],[[0,0,220]],[[0,0,216]],[[0,0,212]],[[0,0,208]],[[0,0,204]],[[0,0,200]],[[0,0,196]],[[0,0,192]],[[0,0,188]],[[0,0,184]],[[0,0,180]],[[0,0,176]],[[0,0,172]],[[0,0,168]],[[0,0,164]],[[0,0,160]],[[0,0,156]],[[0,0,152]],[[0,0,148]],[[0,0,144]],[[0,0,140]],[[0,0,136]],[[0,0,132]],[[0,0,128]]], dtype=np.uint8))
    wls_filter: bool = False
    wls_lambda: float = 8000
    wls_sigma: float = 1.5

@dataclass
class DetectionConfig:
    """Configuration for drawing bounding boxes."""
    thickness: int = 1
    fill_transparency: float = 0.15
    box_roundness: int = 0
    color: Tuple[int, int, int] = (0, 255, 0)
    bbox_style: BboxStyle = BboxStyle.RECTANGLE
    line_width: float = 0.5
    line_height: float = 0.5
    hide_label: bool = False
    label_position: TextPosition = TextPosition.TOP_LEFT
    label_padding: int = 10


@dataclass
class TextConfig:
    """Configuration for drawing labels."""
    font_face: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    font_color: Tuple[int, int, int] = (255, 255, 255)
    font_transparency: float = 0.5
    font_scale: float = 1.0
    font_thickness: int = 2
    font_position: TextPosition = TextPosition.TOP_LEFT

    background_color: Optional[Tuple[int, int, int]] = None
    background_transparency: float = 0.5

    outline_color: Tuple[int, int, int] = (0, 0, 0)

    line_type: int = 16  # cv2.LINE_AA

    auto_scale: bool = True


@dataclass
class TrackingConfig:
    """Configuration for drawing tracking bounding boxes."""
    max_length: int = -1
    deletion_lost_threshold: int = 5
    line_thickness: int = 1
    fading_tails: bool = False
    line_color: Tuple[int, int, int] = (255, 255, 255)
    line_type: int = 16  # cv2.LINE_AA
    show_speed: bool = False


@dataclass
class SegmentationConfig:
    """Configuration for drawing segmentation masks."""
    mask_alpha: float = 0.5


@dataclass
class CircleConfig:
    """Configuration for drawing circles."""
    thickness: int = 1
    color: Tuple[int, int, int] = (255, 255, 255)
    line_type: int = 16  # cv2.LINE_AA


@dataclass
class VisConfig:
    """Configuration for visualizer."""

    output: OutputConfig = field(default_factory=OutputConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    circle: CircleConfig = field(default_factory=CircleConfig)

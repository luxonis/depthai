import math
from enum import IntEnum
from types import SimpleNamespace
from typing import Tuple, Union, List, Any, Dict

try:
    import cv2
except ImportError:
    cv2 = None

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import (
    DetectionPacket,
    _TwoStageDetection,
    SpatialBbMappingPacket,
    TrackerPacket,
    _TrackingDetection
)
from depthai_sdk.visualize.bbox import BoundingBox

class FramePosition(IntEnum):
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


if cv2 is not None:
    default_color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
    default_color_map[0] = [0, 0, 0]
else:
    default_color_map = None


class VisualizerHelper:
    bg_color = (0, 0, 0)
    front_color = (255, 255, 255)
    text_type = 0  # cv2.FONT_HERSHEY_SIMPLEX
    line_type = 16  # cv2.LINE_AA

    @classmethod
    def putText(cls,
                frame: np.ndarray,
                text: str,
                coords: Tuple[int, int],
                scale: float = 1.0,
                backColor: Tuple[int, int, int] = None,
                color: Tuple[int, int, int] = None):
        # Background text

        cv2.putText(frame, text, coords, cls.text_type, scale,
                    color=backColor if backColor else cls.bg_color,
                    thickness=int(scale * 3),
                    lineType=cls.line_type)
        # Front text
        cv2.putText(frame, text, coords, cls.text_type, scale,
                    color=(int(color[0]), int(color[1]), int(color[2])) if color else cls.front_color,
                    thickness=int(scale),
                    lineType=cls.line_type)

    @classmethod
    def line(cls,
             frame: np.ndarray,
             p1: Tuple[int, int], p2: Tuple[int, int],
             color=None,
             thickness: int = 1) -> None:
        cv2.line(frame, p1, p2,
                 cls.bg_color,
                 thickness * 3,
                 cls.line_type)
        cv2.line(frame, p1, p2,
                 (int(color[0]), int(color[1]), int(color[2])) if color else cls.front_color,
                 thickness,
                 cls.line_type)

    @classmethod
    def print_on_roi(cls, frame, topLeft, bottomRight, text: str, position: FramePosition = FramePosition.BottomLeft,
                     padPx=10):
        frame_roi = frame[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        cls.print(frame=frame_roi, text=text, position=position, padPx=padPx)

    @classmethod
    def print(cls, frame, text: str, position: FramePosition = FramePosition.BottomLeft, padPx=10):
        """
        Prints text on the frame.
        @param frame: Frame
        @param text: Text to be printed
        @param position: Where on frame we want to print the text
        @param padPx: Padding (in pixels)
        """
        text_size = cv2.getTextSize(text, VisualizerHelper.text_type, fontScale=1.0, thickness=1)[0]
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        y_pos = int(position) % 10
        if y_pos == 0:  # Y Top
            y = text_size[1] + padPx
        elif y_pos == 1:  # Y Mid
            y = int(frame_h / 2) + int(text_size[1] / 2)
        else:  # y_pos == 2. Y Bottom
            y = frame_h - padPx

        x_pos = int(position) // 10
        if x_pos == 0:  # X Left
            x = padPx
        elif x_pos == 1:  # X Mid
            x = int(frame_w / 2) - int(text_size[0] / 2)
        else:  # x_pos == 2  # X Right
            x = frame_w - text_size[0] - padPx
        cls.putText(frame, text, (x, y))


# def rectangle(frame, bbox, color: Tuple[int, int, int] = None):
#     x1, y1, x2, y2 = bbox
#     cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, 3)
#     cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2),
#                   color=(int(color[0]), int(color[1]), int(color[2])) if color else front_color, thickness=1)


def rectangle(src,
              bbox: np.ndarray,
              color: Tuple[int, int, int],
              thickness=-1,
              radius=0.1,
              line_type=16,  # cv2.LINE_AA
              alpha=0.15):
    """
    Draw the rectangle (bounding box) on the frame.
    @param src: Frame
    @param top_left: Top left corner of the bounding box
    @param bottom_right: Bottom right corner of the bounding box
    @param color: Color of the rectangle
    @param thickness: Thickness of the rectangle. If -1, it will colorize the rectangle as well (with alpha)
    @param radius: Radius of the corners (for rounded rectangle)
    @param line_type: Line type for the rectangle
    @param alpha: Alpha for colorizing of the rectangle
    @return: Frame
    """
    top_left = (bbox[0], bbox[1])
    bottom_right = (bbox[2], bbox[3])
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    height = abs(bottom_right[1] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height / 2))

    if thickness < 0:
        overlay = src.copy()

        # big rect
        top_left_main_rect = (int(top_left[0] + corner_radius), int(top_left[1]))
        bottom_right_main_rect = (int(bottom_right[0] - corner_radius), int(bottom_right[1]))

        top_left_rect_left = (top_left[0], top_left[1] + corner_radius)
        bottom_right_rect_left = (bottom_left[0] + corner_radius, bottom_left[1] - corner_radius)

        top_left_rect_right = (top_right[0] - corner_radius, top_right[1] + corner_radius)
        bottom_right_rect_right = (bottom_right[0], bottom_right[1] - corner_radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right]
        ]

        [cv2.rectangle(overlay, pt1=rect[0], pt2=rect[1], color=color, thickness=thickness) for rect in all_rects]

        cv2.ellipse(overlay, (top_left[0] + corner_radius, top_left[1] + corner_radius), (corner_radius, corner_radius),
                    180.0, 0,
                    90, color, thickness, line_type)
        cv2.ellipse(overlay, (top_right[0] - corner_radius, top_right[1] + corner_radius),
                    (corner_radius, corner_radius),
                    270.0, 0,
                    90, color, thickness, line_type)
        cv2.ellipse(overlay, (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius),
                    (corner_radius, corner_radius), 0.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(overlay, (bottom_left[0] + corner_radius, bottom_left[1] - corner_radius),
                    (corner_radius, corner_radius), 90.0, 0,
                    90, color, thickness, line_type)

        cv2.ellipse(src, (top_left[0] + corner_radius, top_left[1] + corner_radius), (corner_radius, corner_radius),
                    180.0, 0, 90,
                    color, 2, line_type)
        cv2.ellipse(src, (top_right[0] - corner_radius, top_right[1] + corner_radius), (corner_radius, corner_radius),
                    270.0, 0, 90,
                    color, 2, line_type)
        cv2.ellipse(src, (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius),
                    (corner_radius, corner_radius), 0.0, 0, 90,
                    color, 2, line_type)
        cv2.ellipse(src, (bottom_left[0] + corner_radius, bottom_left[1] - corner_radius),
                    (corner_radius, corner_radius),
                    90.0, 0, 90,
                    color, 2, line_type)

        cv2.addWeighted(overlay, alpha, src, 1 - alpha, 0, src)
    else:  # Don't fill the rectangle
        # draw straight lines
        cv2.line(src, (top_left[0] + corner_radius, top_left[1]), (top_right[0] - corner_radius, top_right[1]), color,
                 abs(thickness), line_type)
        cv2.line(src, (top_right[0], top_right[1] + corner_radius), (bottom_right[0], bottom_right[1] - corner_radius),
                 color, abs(thickness), line_type)
        cv2.line(src, (bottom_right[0] - corner_radius, bottom_left[1]),
                 (bottom_left[0] + corner_radius, bottom_right[1]),
                 color, abs(thickness), line_type)
        cv2.line(src, (bottom_left[0], bottom_left[1] - corner_radius), (top_left[0], top_left[1] + corner_radius),
                 color,
                 abs(thickness), line_type)

        # draw arcs
        cv2.ellipse(src, (top_left[0] + corner_radius, top_left[1] + corner_radius), (corner_radius, corner_radius),
                    180.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(src, (top_right[0] - corner_radius, top_right[1] + corner_radius), (corner_radius, corner_radius),
                    270.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(src, (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius),
                    (corner_radius, corner_radius), 0.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(src, (bottom_left[0] + corner_radius, bottom_left[1] - corner_radius),
                    (corner_radius, corner_radius),
                    90.0, 0, 90,
                    color, thickness, line_type)

    return src

def draw_mappings(packet: SpatialBbMappingPacket):
    dets = packet.spatials.detections
    for det in dets:
        roi = det.boundingBoxMapping.roi
        roi = roi.denormalize(packet.frame.shape[1], packet.frame.shape[0])
        top_left = roi.topLeft()
        bottom_right = roi.bottomRight()
        x_min = int(top_left.x)
        y_min = int(top_left.y)
        x_max = int(bottom_right.x)
        y_max = int(bottom_right.y)

        cv2.rectangle(packet.frame, (x_min, y_min), (x_max, y_max), VisualizerHelper.bg_color, 3)
        cv2.rectangle(packet.frame, (x_min, y_min), (x_max, y_max), VisualizerHelper.front_color, 1)


def spatials_text(spatials: dai.Point3f):
    return SimpleNamespace(
        x="X: " + ("{:.1f}m".format(spatials.x / 1000) if not math.isnan(spatials.x) else "--"),
        y="Y: " + ("{:.1f}m".format(spatials.y / 1000) if not math.isnan(spatials.y) else "--"),
        z="Z: " + ("{:.1f}m".format(spatials.z / 1000) if not math.isnan(spatials.z) else "--"),
    )


def draw_detections(packet: Union[DetectionPacket, _TwoStageDetection, TrackerPacket],
                    norm: BoundingBox,
                    label_map: List[Tuple[str, Tuple]] = None):
    """
    Draw object detections to the frame.

    @param frame: np.ndarray frame
    @param dets: dai.ImgDetections
    @param norm: Object that handles normalization of the bounding box
    @param label_map: Label map for the detections
    """
    img_detections = []
    if isinstance(packet, TrackerPacket):
        img_detections = [t.srcImgDetection for t in packet.daiTracklets.tracklets]
    elif isinstance(packet, DetectionPacket):
        img_detections = [det for det in packet.img_detections.detections]

    for detection in img_detections:
        bbox = norm.get_relative_bbox(BoundingBox(detection)).to_tuple(packet.frame.shape)

        if label_map:
            txt, color = label_map[detection.label]
        else:
            txt = str(detection.label)
            color = VisualizerHelper.front_color

        VisualizerHelper.putText(packet.frame, txt, (bbox[0] + 5, bbox[1] + 25), scale=0.9)
        if packet._is_spatial_detection():
            point = packet._get_spatials(detection) \
                if isinstance(packet, TrackerPacket) \
                else detection.spatialCoordinates
            VisualizerHelper.putText(packet.frame, spatials_text(point).x, (bbox[0] + 5, bbox[1] + 50), scale=0.7)
            VisualizerHelper.putText(packet.frame, spatials_text(point).y, (bbox[0] + 5, bbox[1] + 75), scale=0.7)
            VisualizerHelper.putText(packet.frame, spatials_text(point).z, (bbox[0] + 5, bbox[1] + 100), scale=0.7)

        rectangle(packet.frame, bbox, color=color, thickness=1, radius=0)
        packet._add_detection(detection, bbox, txt, color)


def draw_tracklet_id(packet: TrackerPacket):
    for det in packet.detections:
        # centroid = det.centroid()
        VisualizerHelper.print_on_roi(packet.frame, det.top_left, det.bottom_right,
                                      f"Id: {str(det.tracklet.id)}",
                                      FramePosition.TopMid)


def draw_breadcrumb_trail(packets: List[TrackerPacket]):
    packet = packets[-1]  # Current packet

    dict_: Dict[str, List[_TrackingDetection]] = {}
    valid_ids = [t.id for t in packet.daiTracklets.tracklets]
    for idx in valid_ids:
        dict_[str(idx)] = []

    for packet in packets:
        for det in packet.detections:
            if det.tracklet.id in valid_ids:
                dict_[str(det.tracklet.id)].append(det)

    for idx, list_ in dict_.items():
        for i in range(len(list_) - 1):
            VisualizerHelper.line(packet.frame, list_[i].centroid(), list_[i + 1].centroid(), color=list_[i].color)


def colorize_depth(depth_frame: Union[dai.ImgFrame, Any], color_map=None):
    if isinstance(depth_frame, dai.ImgFrame):
        depth_frame = depth_frame.getFrame()

    depth_frame_color = cv2.normalize(depth_frame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
    depth_frame_color = cv2.equalizeHist(depth_frame_color)

    return cv2.applyColorMap(depth_frame_color, color_map or default_color_map)


def colorize_disparity(frame: Union[dai.ImgFrame, Any], multiplier: float, color_map=None):
    if isinstance(frame, dai.ImgFrame):
        frame = frame.getFrame()

    frame = (frame * multiplier).astype(np.uint8)

    return cv2.applyColorMap(frame, color_map or default_color_map)


def draw_bb_mappings(depth_frame: Union[dai.ImgFrame, Any], bb_mappings: dai.SpatialLocationCalculatorConfig):
    depth_frame_color = colorize_depth(depth_frame)
    roi_datas = bb_mappings.getConfigData()
    for roi_data in roi_datas:
        roi = roi_data.roi
        roi = roi.denormalize(depth_frame_color.shape[1], depth_frame_color.shape[0])
        top_left = roi.topLeft()
        bottom_right = roi.bottomRight()
        xmin = int(top_left.x)
        ymin = int(top_left.y)
        xmax = int(bottom_right.x)
        ymax = int(bottom_right.y)

        rectangle(depth_frame_color, (xmin, ymin, xmax, ymax), (255, 255, 255), 1)


def depth_to_disp_factor(device: dai.Device, stereo: dai.node.StereoDepth) -> float:
    """
    Calculates the disparity factor used to calculate disparity from depth, which is used for visualization.
    `disparity[0..95] = disparity_factor / depth`. We can then multiply disparity by 255/95 to get 0..255 range.
    @param device: OAK device
    """
    calib = device.readCalibration()
    cam1=calib.getStereoLeftCameraId()
    cam2=calib.getStereoRightCameraId()
    baseline = calib.getBaselineDistance(cam1=cam1, cam2=cam2, useSpecTranslation=True) * 10  # cm to mm
    rawConf = stereo.initialConfig.get()

    align: dai.CameraBoardSocket = stereo.properties.depthAlignCamera
    if align == dai.CameraBoardSocket.AUTO:
        align = cam2

    intrinsics = calib.getCameraIntrinsics(align)
    focalLength = intrinsics[0][0]

    factor = baseline * focalLength
    if rawConf.algorithmControl.enableExtended:
        factor /= 2

    return factor

def hex_to_bgr(hex: str) -> Tuple[int, ...]:
    """
    "#ff1f00" (red) => (0, 31, 255)
    """
    value = hex.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))

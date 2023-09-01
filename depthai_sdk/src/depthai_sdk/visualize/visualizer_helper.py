from enum import IntEnum
from typing import Tuple, Union, List, Any, Dict

from depthai_sdk.classes.nn_results import TrackingDetection, TwoStageDetection
from depthai_sdk.visualize.configs import BboxStyle
from depthai_sdk.visualize.objects import VisBoundingBox

try:
    import cv2
except ImportError:
    cv2 = None

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import (
    DetectionPacket,
    SpatialBbMappingPacket,
    TrackerPacket,
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


def draw_detections(packet: Union[DetectionPacket, TwoStageDetection, TrackerPacket],
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

    dict_: Dict[str, List[TrackingDetection]] = {}
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
    cam1 = calib.getStereoLeftCameraId()
    cam2 = calib.getStereoRightCameraId()
    baseline = calib.getBaselineDistance(cam1=cam1, cam2=cam2, useSpecTranslation=True) * 10  # cm to mm
    raw_conf = stereo.initialConfig.get()

    align: dai.CameraBoardSocket = stereo.properties.depthAlignCamera
    if align == dai.CameraBoardSocket.AUTO:
        align = cam2

    intrinsics = calib.getCameraIntrinsics(align)
    focal_length = intrinsics[0][0]

    factor = baseline * focal_length
    if raw_conf.algorithmControl.enableExtended:
        factor /= 2

    return factor


def draw_bbox(img: np.ndarray,
              pt1: Tuple[int, int],
              pt2: Tuple[int, int],
              color: Tuple[int, int, int],
              thickness: int,
              r: int,
              line_width: int,
              line_height: int,
              alpha: float
              ) -> None:
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
        alpha: Opacity of the rectangle.
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
    if 0 < alpha:
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


def draw_stylized_bbox(img: np.ndarray, obj: VisBoundingBox) -> None:
    """
    Draw a stylized bounding box. The style is either passed as an argument or defined in the config.

    Args:
        img: Image to draw on.
        obj: Bounding box to draw.
    """
    pt1, pt2 = obj.bbox.denormalize(img.shape)

    box_w = pt2[0] - pt1[0]
    box_h = pt2[1] - pt1[1]

    line_width = int(box_w * obj.config.detection.line_width) // 2
    line_height = int(box_h * obj.config.detection.line_height) // 2
    roundness = int(obj.config.detection.box_roundness)
    bbox_style = obj.bbox_style or obj.config.detection.bbox_style
    alpha = obj.config.detection.fill_transparency

    if bbox_style == BboxStyle.RECTANGLE:
        draw_bbox(img, pt1, pt2,
                  obj.color, obj.thickness, 0,
                  line_width=0, line_height=0, alpha=alpha)
    elif bbox_style == BboxStyle.CORNERS:
        draw_bbox(img, pt1, pt2,
                  obj.color, obj.thickness, 0,
                  line_width=line_width, line_height=line_height, alpha=alpha)
    elif bbox_style == BboxStyle.ROUNDED_RECTANGLE:
        draw_bbox(img, pt1, pt2,
                  obj.color, obj.thickness, roundness,
                  line_width=0, line_height=0, alpha=alpha)
    elif bbox_style == BboxStyle.ROUNDED_CORNERS:
        draw_bbox(img, pt1, pt2,
                  obj.color, obj.thickness, roundness,
                  line_width=line_width, line_height=line_height, alpha=alpha)

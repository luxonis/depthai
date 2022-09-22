import numpy as np
import depthai as dai
from typing import Tuple, Union, List, Any, Callable
import cv2
import distinctipy
from .normalize_bb import NormalizeBoundingBox
from ..classes.packets import DetectionPacket, TwoStageDetection



bg_color = (0, 0, 0)
front_color = (255, 255, 255)
text_type = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA
class Visualizer:
    bg_color = (0, 0, 0)
    front_color = (255, 255, 255)
    text_type = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    @staticmethod
    def putText(frame: np.ndarray,
                text: str,
                coords: Tuple[int,int],
                scale: float = 1.0,
                backColor: Tuple[int,int,int] = None,
                color: Tuple[int, int, int] = None):
        # Background text
        cv2.putText(frame, text, coords, text_type, scale,
                    color=backColor if backColor else bg_color,
                    thickness=int(scale * 3),
                    lineType=line_type)
        # Front text
        cv2.putText(frame, text, coords, text_type, scale,
                    color=(int(color[0]), int(color[1]), int(color[2])) if color else front_color,
                    thickness=int(scale),
                    lineType=line_type)


# def rectangle(frame, bbox, color: Tuple[int, int, int] = None):
#     x1, y1, x2, y2 = bbox
#     cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, 3)
#     cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2),
#                   color=(int(color[0]), int(color[1]), int(color[2])) if color else front_color, thickness=1)


def rectangle(src,
              bbox: np.ndarray,
              color: Tuple[int,int,int],
              thickness=-1,
              radius=0.1,
              line_type=cv2.LINE_AA,
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
    topLeft = (bbox[0], bbox[1])
    bottomRight = (bbox[2], bbox[3])
    topRight = (bottomRight[0], topLeft[1])
    bottomLeft = (topLeft[0], bottomRight[1])

    height = abs(bottomRight[1] - topLeft[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height / 2))

    if thickness < 0:
        overlay = src.copy()

        # big rect
        top_left_main_rect = (int(topLeft[0] + corner_radius), int(topLeft[1]))
        bottom_right_main_rect = (int(bottomRight[0] - corner_radius), int(bottomRight[1]))

        top_left_rect_left = (topLeft[0], topLeft[1] + corner_radius)
        bottom_right_rect_left = (bottomLeft[0] + corner_radius, bottomLeft[1] - corner_radius)

        top_left_rect_right = (topRight[0] - corner_radius, topRight[1] + corner_radius)
        bottom_right_rect_right = (bottomRight[0], bottomRight[1] - corner_radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(overlay, pt1=rect[0], pt2=rect[1], color=color, thickness=thickness) for rect in all_rects]

        cv2.ellipse(overlay, (topLeft[0] + corner_radius, topLeft[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0,
                    90, color, thickness, line_type)
        cv2.ellipse(overlay, (topRight[0] - corner_radius, topRight[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0,
                    90, color, thickness, line_type)
        cv2.ellipse(overlay, (bottomRight[0] - corner_radius, bottomRight[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(overlay, (bottomLeft[0] + corner_radius, bottomLeft[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0,
                    90, color, thickness, line_type)

        cv2.ellipse(src, (topLeft[0] + corner_radius, topLeft[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90,
                    color, 2, line_type)
        cv2.ellipse(src, (topRight[0] - corner_radius, topRight[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90,
                    color, 2, line_type)
        cv2.ellipse(src, (bottomRight[0] - corner_radius, bottomRight[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,
                    color, 2, line_type)
        cv2.ellipse(src, (bottomLeft[0] + corner_radius, bottomLeft[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,
                    color, 2, line_type)

        cv2.addWeighted(overlay, alpha, src, 1 - alpha, 0, src)
    else:  # Don't fill the rectangle
        # draw straight lines
        cv2.line(src, (topLeft[0] + corner_radius, topLeft[1]), (topRight[0] - corner_radius, topRight[1]), color, abs(thickness), line_type)
        cv2.line(src, (topRight[0], topRight[1] + corner_radius), (bottomRight[0], bottomRight[1] - corner_radius), color, abs(thickness), line_type)
        cv2.line(src, (bottomRight[0] - corner_radius, bottomLeft[1]), (bottomLeft[0] + corner_radius, bottomRight[1]), color, abs(thickness), line_type)
        cv2.line(src, (bottomLeft[0], bottomLeft[1] - corner_radius), (topLeft[0], topLeft[1] + corner_radius), color, abs(thickness), line_type)

        # draw arcs
        cv2.ellipse(src, (topLeft[0] + corner_radius, topLeft[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(src, (topRight[0] - corner_radius, topRight[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(src, (bottomRight[0] - corner_radius, bottomRight[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,
                    color, thickness, line_type)
        cv2.ellipse(src, (bottomLeft[0] + corner_radius, bottomLeft[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,
                    color, thickness, line_type)

    return src


def get_text_color(background, threshold=0.6):
    bck = np.array(background) / 256
    clr = distinctipy.get_text_color((bck[2], bck[1], bck[0]), threshold)
    clr = distinctipy.get_rgb256(clr)
    return (clr[2], clr[1], clr[0])


def drawDetections(packet: Union[DetectionPacket, TwoStageDetection],
                   norm: NormalizeBoundingBox,
                   labelMap: List[Tuple[str, Tuple]] = None):
    """
    Draw object detections to the frame.

    @param frame: np.ndarray frame
    @param dets: dai.ImgDetections
    @param norm: Object that handles normalization of the bounding box
    @param labelMap: Label map for the detections
    @param callback: Callback that will be called on each object, with (frame, bbox) in arguments
    """
    for detection in packet.imgDetections.detections:
        bbox = norm.normalize(packet.frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

        if labelMap:
            txt, color = labelMap[detection.label]
        else:
            txt = str(detection.label)
            color = front_color

        Visualizer.putText(packet.frame, txt, (bbox[0] + 5, bbox[1] + 25), scale=0.9)
        if packet.isSpatialDetection():
            Visualizer.putText(packet.frame, packet.spatialsText(detection).x, (bbox[0] + 5, bbox[1] + 50), scale=0.7)
            Visualizer.putText(packet.frame, packet.spatialsText(detection).y, (bbox[0] + 5, bbox[1] + 75), scale=0.7)
            Visualizer.putText(packet.frame, packet.spatialsText(detection).z, (bbox[0] + 5, bbox[1] + 100), scale=0.7)

        rectangle(packet.frame, bbox, color=color, thickness=1, radius=0)

        packet.add_detection(detection, bbox, txt, color)


jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
jet_custom = jet_custom[::-1]
jet_custom[0] = [0, 0, 0]


def colorizeDepth(depthFrame: Union[dai.ImgFrame, Any], colorMap=None):
    if isinstance(depthFrame, dai.ImgFrame):
        depthFrame = depthFrame.getFrame()
    depthFrameColor = cv2.normalize(depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    return cv2.applyColorMap(depthFrameColor, colorMap if colorMap else jet_custom)

def colorizeDisparity(depthFrame: Union[dai.ImgFrame, Any],  colorMap=None):
    if isinstance(depthFrame, dai.ImgFrame):
        depthFrame = depthFrame.getFrame()
    depthFrameColor = cv2.normalize(depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    return cv2.applyColorMap(depthFrameColor, colorMap if colorMap else jet_custom)

def drawBbMappings(depthFrame: Union[dai.ImgFrame, Any], bbMappings: dai.SpatialLocationCalculatorConfig):
    depthFrameColor = colorizeDepth(depthFrame)
    roiDatas = bbMappings.getConfigData()
    for roiData in roiDatas:
        roi = roiData.roi
        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        xmin = int(topLeft.x)
        ymin = int(topLeft.y)
        xmax = int(bottomRight.x)
        ymax = int(bottomRight.y)

        rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 255, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


def hex_to_bgr(hex: str) -> Tuple[int, int, int]:
    """
    "#ff1f00" (red) => (0, 31, 255)
    """
    value = hex.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))
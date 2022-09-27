import math
import time
from enum import IntEnum
from types import SimpleNamespace
import numpy as np
import depthai as dai
from typing import Tuple, Union, List, Any, Callable, Dict
import cv2
import distinctipy
from .normalize_bb import NormalizeBoundingBox
from ..classes.packets import DetectionPacket, TwoStageDetection, FramePacket, SpatialBbMappingPacket, TrackerPacket, \
    TrackingDetection


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

color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
color_map[0] = [0, 0, 0]

class Visualizer:
    bg_color = (0, 0, 0)
    front_color = (255, 255, 255)
    text_type = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    @classmethod
    def putText(cls, frame: np.ndarray,
                text: str,
                coords: Tuple[int,int],
                scale: float = 1.0,
                backColor: Tuple[int,int,int] = None,
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
    def line(cls, frame: np.ndarray,
             p1: Tuple[int,int], p2: Tuple[int,int],
             color = None,
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
    def print_on_roi(cls, frame, topLeft, bottomRight, text:str, position: FramePosition = FramePosition.BottomLeft, padPx=10):
        frame_roi = frame[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
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
        textSize = cv2.getTextSize(text, Visualizer.text_type, fontScale=1.0, thickness=1)[0]
        frameW = frame.shape[1]
        frameH = frame.shape[0]

        yPos = int(position) % 10
        if yPos == 0:  # Y Top
            y = textSize[1] + padPx
        elif yPos == 1:  # Y Mid
            y = int(frameH / 2) + int(textSize[1] / 2)
        else:  # yPos == 2. Y Bottom
            y = frameH - padPx

        xPos = int(position) // 10
        if xPos == 0:  # X Left
            x = padPx
        elif xPos == 1:  # X Mid
            x = int(frameW / 2) - int(textSize[0] / 2)
        else:  # xPos == 2  # X Right
            x = frameW - textSize[0] - padPx
        cls.putText(frame, text, (x, y))

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


def drawMappings(packet: SpatialBbMappingPacket):
    roiDatas = packet.config.getConfigData()
    for roiData in roiDatas:
        roi = roiData.roi
        roi = roi.denormalize(packet.frame.shape[1], packet.frame.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        xmin = int(topLeft.x)
        ymin = int(topLeft.y)
        xmax = int(bottomRight.x)
        ymax = int(bottomRight.y)

        cv2.rectangle(packet.frame, (xmin, ymin), (xmax, ymax), Visualizer.bg_color, 3)
        cv2.rectangle(packet.frame, (xmin, ymin), (xmax, ymax), Visualizer.front_color, 1)

def spatialsText(spatials: dai.Point3f):
    return SimpleNamespace(
        x = "X: " + ("{:.1f}m".format(spatials.x / 1000) if not math.isnan(spatials.x) else "--"),
        y = "Y: " + ("{:.1f}m".format(spatials.y / 1000) if not math.isnan(spatials.y) else "--"),
        z = "Z: " + ("{:.1f}m".format(spatials.z / 1000) if not math.isnan(spatials.z) else "--"),
    )

def drawDetections(packet: Union[DetectionPacket, TwoStageDetection, TrackerPacket],
                   norm: NormalizeBoundingBox,
                   labelMap: List[Tuple[str, Tuple]] = None):
    """
    Draw object detections to the frame.

    @param frame: np.ndarray frame
    @param dets: dai.ImgDetections
    @param norm: Object that handles normalization of the bounding box
    @param labelMap: Label map for the detections
    """
    imgDets = []
    if isinstance(packet, TrackerPacket):
        imgDets = [t.srcImgDetection for t in packet.daiTracklets.tracklets]
    elif isinstance(packet, DetectionPacket):
        imgDets = [det for det in packet.imgDetections.detections]

    for detection in imgDets:
        bbox = norm.normalize(packet.frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

        if labelMap:
            txt, color = labelMap[detection.label]
        else:
            txt = str(detection.label)
            color = Visualizer.front_color

        Visualizer.putText(packet.frame, txt, (bbox[0] + 5, bbox[1] + 25), scale=0.9)
        if packet.isSpatialDetection():
            point = packet.getSpatials(detection) if isinstance(packet, TrackerPacket) else detection.spatialCoordinates
            Visualizer.putText(packet.frame, spatialsText(point).x, (bbox[0] + 5, bbox[1] + 50), scale=0.7)
            Visualizer.putText(packet.frame, spatialsText(point).y, (bbox[0] + 5, bbox[1] + 75), scale=0.7)
            Visualizer.putText(packet.frame, spatialsText(point).z, (bbox[0] + 5, bbox[1] + 100), scale=0.7)

        rectangle(packet.frame, bbox, color=color, thickness=1, radius=0)
        packet.add_detection(detection, bbox, txt, color)

def drawTrackletId(packet: TrackerPacket):
    for det in packet.detections:
        centroid = det.centroid()
        Visualizer.print_on_roi(packet.frame, det.topLeft, det.bottomRight,
                                f"Id: {str(det.tracklet.id)}",
                                FramePosition.TopMid)
def drawBreadcrumbTrail(packets: List[TrackerPacket]):
    packet = packets[-1] # Current packet

    dic: Dict[str, List[TrackingDetection]] = {}
    validIds = [t.id for t in packet.daiTracklets.tracklets]
    for id in validIds:
        dic[str(id)] = []

    for packet in packets:
        for det in packet.detections:
            if det.tracklet.id in validIds:
                dic[str(det.tracklet.id)].append(det)

    for id, list in dic.items():
        for i in range(len(list) - 1):
            Visualizer.line(packet.frame, list[i].centroid(), list[i+1].centroid(), color=list[i].color)


def colorizeDepth(depthFrame: Union[dai.ImgFrame, Any], colorMap=None):
    if isinstance(depthFrame, dai.ImgFrame):
        depthFrame = depthFrame.getFrame()
    depthFrameColor = cv2.normalize(depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    return cv2.applyColorMap(depthFrameColor, colorMap if colorMap else color_map)

def colorizeDisparity(frame: Union[dai.ImgFrame, Any], multiplier: float, colorMap=None):
    if isinstance(frame, dai.ImgFrame):
        frame = frame.getFrame()
    frame = (frame * multiplier).astype(np.uint8)
    return cv2.applyColorMap(frame, colorMap if colorMap else color_map)

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

def calc_disp_multiplier(device: dai.Device, size: Tuple[int,int]) -> float:
    calib = device.readCalibration()
    baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, size)
    focalLength = intrinsics[0][0]
    return baseline * focalLength

def hex_to_bgr(hex: str) -> Tuple[int, int, int]:
    """
    "#ff1f00" (red) => (0, 31, 255)
    """
    value = hex.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))
import numpy as np
import depthai as dai
from typing import Tuple, Union, List, Any, Callable
import cv2
import distinctipy
from .. import AspectRatioResizeMode


bg_color = (0, 0, 0)
front_color = (255, 255, 255)
text_type = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA


def putText(frame, text, coords, scale: float = 1.0, backColor = None, color: Tuple[int, int, int] = None):
    cv2.putText(frame, text, coords, text_type, scale, backColor if backColor else bg_color, int(scale * 3), line_type)
    cv2.putText(frame, text=text, org=coords, fontFace=text_type, fontScale=scale,
                color=(int(color[0]), int(color[1]), int(color[2])) if color else front_color, thickness=int(scale),
                lineType=line_type)


# def rectangle(frame, bbox, color: Tuple[int, int, int] = None):
#     x1, y1, x2, y2 = bbox
#     cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, 3)
#     cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2),
#                   color=(int(color[0]), int(color[1]), int(color[2])) if color else front_color, thickness=1)


def rectangle(src,
              bbox,
              color,
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


class NormalizeBoundingBox:
    def __init__(self,
                 aspectRatio: Tuple[float, float],
                 arResizeMode: AspectRatioResizeMode,
                 ):
        """
        @param aspectRatio: NN input size
        @param arResizeMode
        """
        self.aspectRatio = aspectRatio
        self.arResizeMode = arResizeMode

    def normalize(self, frame, bbox: Tuple[float, float, float, float]):
        """
        Mapps bounding box coordinates (0..1) to pixel values on frame

        Args:
            frame (numpy.ndarray): Frame to which adjust the bounding box
            bbox (list): list of bounding box points in a form of :code:`[x1, y1, x2, y2, ...]`

        Returns:
            list: Bounding box points mapped to pixel values on frame
        """
        bbox = np.array(bbox)

        # Edit the bounding boxes before normalizing them
        if self.arResizeMode == AspectRatioResizeMode.CROP:
            ar_diff = self.aspectRatio[0] / self.aspectRatio[1] - frame.shape[1] / frame.shape[0]
            sel = 0 if 0 < ar_diff else 1
            bbox[sel::2] *= 1 - abs(ar_diff)
            bbox[sel::2] += abs(ar_diff) / 2
        elif self.arResizeMode == AspectRatioResizeMode.STRETCH:
            # No need to edit bounding boxes when stretching
            pass
        elif self.arResizeMode == AspectRatioResizeMode.LETTERBOX:
            # There might be better way of doing this. TODO: test if it works as expected
            ar_diff = self.aspectRatio[0] / self.aspectRatio[1] - frame.shape[1] / frame.shape[0]
            sel = 0 if 0 < ar_diff else 1
            nsel = 0 if sel == 1 else 1
            # Get the divisor
            div = frame.shape[sel] / self.aspectRatio[nsel]
            letterboxing_ratio = 1 - (frame.shape[nsel] / div) / self.aspectRatio[sel]

            bbox[sel::2] -= abs(letterboxing_ratio) / 2
            bbox[sel::2] /= 1 - abs(letterboxing_ratio)

        # Normalize bounding boxes
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(bbox, 0, 1) * normVals).astype(int)


def get_text_color(background, threshold=0.6):
    bck = np.array(background) / 256
    clr = distinctipy.get_text_color((bck[2], bck[1], bck[0]), threshold)
    clr = distinctipy.get_rgb256(clr)
    return (clr[2], clr[1], clr[0])

def drawDetections(frame,
                   dets: dai.ImgDetections,
                   norm: NormalizeBoundingBox,
                   labelMap: List[Tuple[str, Tuple]] = None,
                   callback: Callable = None):
    """
    Draw object detections to the frame.

    @param frame: np.ndarray frame
    @param dets: dai.ImgDetections
    @param norm: Object that handles normalization of the bounding box
    @param labelMap: Label map for the detections
    @param callback: Callback that will be called on each object, with (frame, bbox) in arguments
    """
    for detection in dets.detections:
        bbox = norm.normalize(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        color, txt = None, None
        if labelMap:
            txt, color = labelMap[detection.label]
        else:
            txt = str(detection.label)
        putText(frame, txt, (bbox[0] + 10, bbox[1] + 20), color=color)
        rectangle(frame, bbox, color=color, thickness=1, radius=0)
        if callback:
            callback(frame, bbox)


jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_custom = jet_custom[::-1]
jet_custom[0] = [0, 0, 0]


def colorizeDepth(depthFrame: Union[dai.ImgFrame, Any], colorMap=None):
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
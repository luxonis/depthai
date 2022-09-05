import numpy as np
import depthai as dai
from typing import Tuple, Optional, Union, List, Dict, Type, Any, Callable
import cv2
import time
import distinctipy
from .. import AspectRatioResizeMode
from ..components import Component, NNComponent
from enum import IntEnum

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
            ar_diff = self.aspectRatio[0] / self.aspectRatio[1] - frame.shape[0] / frame.shape[1]
            sel = 0 if 0 < ar_diff else 1
            bbox[sel::2] *= 1 - abs(ar_diff)
            bbox[sel::2] += abs(ar_diff) / 2
        elif self.arResizeMode == AspectRatioResizeMode.STRETCH:
            # No need to edit bounding boxes when stretching
            pass
        elif self.arResizeMode == AspectRatioResizeMode.LETTERBOX:
            ar_diff = self.aspectRatio[0] / self.aspectRatio[1] - frame.shape[0] / frame.shape[1]
            sel = 1 if 0 < ar_diff else 0
            bbox[sel::2] -= abs(ar_diff) / 2
            bbox[sel::2] /= 1 - abs(ar_diff)
        # Normalize bounding boxes
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(bbox, 0, 1) * normVals).astype(int)


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


def get_text_color(background, threshold=0.6):
    bck = np.array(background) / 256
    clr = distinctipy.get_text_color((bck[2], bck[1], bck[0]), threshold)
    clr = distinctipy.get_rgb256(clr)
    return (clr[2], clr[1], clr[0])

def drawDetections(frame,
                   dets: dai.ImgDetections,
                   norm: NormalizeBoundingBox,
                   labelMap: List[Tuple[str, Tuple]] = None):
    for detection in dets.detections:
        bbox = norm.normalize(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        color, txt = None, None
        if labelMap:
            txt, color = labelMap[detection.label]
        else:
            txt = str(detection.label)

        putText(frame, txt, (bbox[0] + 10, bbox[1] + 20), color=color)
        rectangle(frame, bbox, color=color)


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

class BaseVisualizer:
    _name: str
    _scale: Union[None, float, Tuple[int, int]] = None
    _fps: Dict[str, FPSHandler] = None
    _callback: Callable = None

    def __init__(self, frameStream: str) -> None:
        self._name = frameStream

    def setBase(self,
                scale: Union[None, float, Tuple[int, int]] = None,
                fps: Dict[str, FPSHandler] = None,
                callback: Callable = None
                ):
        self._scale = scale
        self._fps = fps
        self._callback = callback

    def newMsgs(self, input: Union[Dict, dai.ImgFrame]):
        if isinstance(input, Dict):
            frame = input[self._name]
        if isinstance(frame, dai.ImgFrame):
            frame = input.getCvFrame()

        if self._fps:
            i = 0
            for name, handler in self._fps.items():
                putText(frame, "{} FPS: {:.1f}".format(name, handler.fps()), (10, 20 + i * 20), scale=0.7)
                i += 1

        if self._scale:
            if isinstance(self._scale, Tuple):
                frame = cv2.resize(frame, self._scale)  # Resize frame
            elif isinstance(self._scale, float):
                shape = frame.shape
                frame = cv2.resize(frame, (
                    int(frame.shape[1] * self._scale),
                    int(frame.shape[0] * self._scale)
                ))

        if self._callback:  # Don't display frame, call the callback
            self._callback(input, frame)
        else:
            cv2.imshow(self.name, frame)

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def print(frame, text: str, position: FramePosition = FramePosition.BottomLeft, padPx=10):
        """
        Prints text on the frame.
        @param frame: Frame
        @param text: Text to be printed
        @param position: Where on frame we want to print the text
        @param padPx: Padding (in pixels)
        """
        textSize = cv2.getTextSize(text, text_type, fontScale=1.0, thickness=1)[0]
        frameW = frame.shape[1]
        frameH = frame.shape[0]

        yPos = int(position) % 10
        if yPos == 0: # Y Top
            y = textSize[1] + padPx
        elif yPos == 1: # Y Mid
            y = int(frameH / 2) + int(textSize[1] / 2)
        else:  # yPos == 2. Y Bottom
            y = frameH - padPx

        xPos = int(position) // 10
        if xPos == 0: # X Left
            x = padPx
        elif xPos == 1:  # X Mid
            x = int(frameW / 2) - int(textSize[0] / 2)
        else:  # xPos == 2  # X Right
            x = frameW - textSize[0] - padPx

        putText(frame, text, (x, y))



class DetectionsVisualizer(BaseVisualizer):
    detectionStream: str  # Detection stream name
    labels: List[Tuple[str, Tuple[int, int, int]]] = None
    normalizer: NormalizeBoundingBox

    def __init__(self,
                 frameStream: str,
                 detectionsStream: str,
                 nnComp: NNComponent
                 ) -> None:
        """
        Visualizes object detection results.

        Args:
            frameStream (str): Name of the frame stream to which we will draw detection results
            detectionsStream (str): Name of the detections stream
            labels (List, optional): List of mappings for object detection labels. List of either labels, or (label, color)
            aspectRatio (Tuple, optional): Aspect ratio of the input frame object detector
            fps (bool): Whether we want to display FPS on the frame
        """
        super().__init__(frameStream)
        # TODO: add support for colors, generate new colors for each label that doesn't have colors
        if nnComp.labels:
            self.labels = []
            n_colors = [isinstance(label, str) for label in nnComp.labels].count(True)
            # np.array of (b,g,r), 0..1
            colors = np.array(distinctipy.get_colors(n_colors=n_colors, rng=154165170, pastel_factor=0.5))[..., ::-1]
            colors = [distinctipy.get_rgb256(clr) for clr in colors]  # List of (b,g,r), 0..255
            for label in nnComp.labels:
                if isinstance(label, str):
                    text = label
                    color = colors.pop(0)  # Take last row
                elif isinstance(label, list):
                    text = label[0]
                    color = hex_to_bgr(label[1])
                else:
                    raise ValueError('Model JSON config error. Label map list can have either str or list!')

                self.labels.append((text, color))

        self.detectionStream = detectionsStream
        self.normalizer = NormalizeBoundingBox(nnComp.size, nnComp.arResizeMode)

    def newMsgs(self, msgs: Dict):
        imgFrame: dai.ImgFrame = msgs[super().name]
        frame = imgFrame.getCvFrame()
        dets = msgs[self.detectionStream]
        drawDetections(frame, dets, self.normalizer, self.labels)
        msgs[super().name] = frame
        super().newMsgs(msgs)


class Visualizer:
    _components: List[Component]
    _visualizers: List[BaseVisualizer] = []
    _scale: Union[None, float, Tuple[int, int]] = None
    _fps: Dict[str, FPSHandler] = None
    _callback: Callable = None

    def __init__(self, components: List[Component],
                 scale: Union[None, float, Tuple[int, int]] = None,
                 fpsHandlers: Dict[str, FPSHandler] = None,
                 callback: Callable = None) -> None:
        self._components = components
        self._scale = scale
        self._fps = fpsHandlers
        self._callback = callback

    def setup(self):
        """
        Called after connected to the device, and all components have been configured
        @return:
        """

        nns = self._components_by_type(self._components, NNComponent)
        frames = self._streams_by_type(self._components, dai.ImgFrame)

        if len(nns) == 0:
            for frame in frames:
                vis = BaseVisualizer(frame)
                vis.setBase(self._scale, self._fps, self._callback)
                self._visualizers.append(vis)
        else:
            for nn in nns:
                for frame in frames:
                    nnStreamNames = self._streams_by_type_xout(nn.xouts, dai.ImgDetections)
                    detVis = DetectionsVisualizer(frame, nnStreamNames[0], nn)
                    detVis.setBase(self._scale, self._fps, self._callback)
                    self._visualizers.append(detVis)

    def _getStreamName(self, xouts: Dict, type: Type) -> str:
        for name, (compType, daiType) in xouts.items():
            if daiType == type: return name
        raise ValueError('Stream name was not found in these Xouts!')

    # Called via callback
    def newMsgs(self, msgs: Dict):
        for vis in self._visualizers:
            vis.newMsgs(msgs)
        # frame = self._getFirstMsg(msgs, dai.ImgFrame).getCvFrame()
        # dets = self._getFirstMsg(msgs, dai.ImgDetections).detections

    def _MsgsList(self, msgs: Dict) -> List[Tuple]:
        arr = []
        for name, msg in msgs:
            arr.append((name, msg, type(msg)))
        return arr

    def _streams_by_type(self, components: List[Component], type: Type) -> List[str]:
        streams = []
        for comp in components:
            for name, (compType, daiType) in comp.xouts.items():
                if daiType == type:
                    streams.append(name)
        return streams

    def _components_by_type(self, components: List[Component], type: Type) -> List[Component]:
        comps = []
        for comp in components:
            if isinstance(comp, type):
                comps.append(comp)
        return comps

    def _getTypeDict(self, msgs: Dict) -> Dict[str, Type]:
        ret = dict()
        for name, msg in msgs:
            ret[name] = type(msg)
        return ret

    def _getComponent(self, streamName: str) -> Component:
        for comp in self._components:
            if streamName in comp.xouts:
                return comp

        raise ValueError("[SDK Visualizer] stream name wasn't found in any component!")
        return

    def _streams_by_type_xout(self, xouts: Dict[str, Tuple[type, type]], type: Type) -> List[str]:
        streams = []
        for name, (compType, daiType) in xouts.items():
            if type == daiType:
                streams.append(name)
        return streams

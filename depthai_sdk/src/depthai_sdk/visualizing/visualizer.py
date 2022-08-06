import numpy as np
import depthai as dai
from typing import Tuple, Optional, Union, List, Dict, Type, Any
import cv2
import time
from ..components import Component, NNComponent

bg_color = (0, 0, 0)
color = (255, 255, 255)
text_type = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA
def putText(frame, text, coords):
    cv2.putText(frame, text, coords, text_type, 1.0, bg_color, 3, line_type)
    cv2.putText(frame, text, coords, text_type, 1.0, color, 1, line_type)
def rectangle(frame, bbox):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

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

def frameNorm(frame, bbox: Tuple[float,float,float,float], aspectRatio: Optional[Tuple[int,int]]= None):
    """
    Mapps bounding box coordinates (0..1) to pixel values on frame

    Args:
        frame (numpy.ndarray): Frame to which adjust the bounding box
        bbox (list): list of bounding box points in a form of :code:`[x1, y1, x2, y2, ...]`
        aspectRatio (Tuple[float,float], optional): Aspect ratio of the NN, for which the image was cropped

    Returns:
        list: Bounding box points mapped to pixel values on frame
    """
    bbox = np.array(bbox)
    if aspectRatio: # Edit bbox values to match that of the NN (that uses cropped image)
        ar_diff = aspectRatio[0] / aspectRatio[1] - frame.shape[0] / frame.shape[1]
        sel = 0 if 0 < ar_diff else 1
        bbox[sel::2] *= 1-abs(ar_diff)
        bbox[sel::2] += abs(ar_diff)/2
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(bbox, 0, 1) * normVals).astype(int)

def drawDetections(frame, dets: dai.ImgDetections, labelMap: List[str] =  None, aspectRatio: Optional[Tuple[float,float]]= None):
    for detection in dets.detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax), aspectRatio)
        putText(frame, labelMap[detection.label] if labelMap else str(detection.label), (bbox[0] + 10, bbox[1] + 20))
        putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40))
        rectangle(frame, bbox)


jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_custom = jet_custom[::-1]
jet_custom[0] = [0, 0, 0]
def colorizeDepth(depthFrame: Union[dai.ImgFrame, Any], colorMap = None):
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

def hex_to_rgb(hex: str) -> Tuple[int,int,int]:
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def hex_to_bgr(hex: str) -> Tuple[int,int,int]:
    rgb = hex_to_rgb(hex)
    return tuple(rgb[2], rgb[1], rgb[0])

class BaseVisualizer():
    _name: str
    _scale: Union[None, float, Tuple[int,int]] = None
    _fps: FPSHandler = None

    def __init__(self, frameStream: str) -> None:
        self._name = frameStream

    def setBase(self,
        scale: Union[None, float, Tuple[int,int]] = None,
        fps: bool = False,
        ):
        self._scale = scale
        if fps:
            self._fps = FPSHandler()

    def newMsgs(self, frame: Union[Dict, dai.ImgFrame, Any]):
        if isinstance(frame, Dict):
            frame = frame[self._name]
        if isinstance(frame, dai.ImgFrame):
            frame = frame.getCvFrame()

        if self._fps:
            self._fps.next_iter()
            putText(frame, "FPS: {:.1f}".format(self._fps.fps()), (10,20))

        if self._scale:
            if isinstance(self._scale, Tuple):
                frame = cv2.resize(frame, self._scale) # Resize frame
            elif isinstance(self._scale, float):
                shape = frame.shape
                frame = cv2.resize(frame, (
                    int(frame.shape[1] * self._scale),
                    int(frame.shape[0] * self._scale)
                ))

        cv2.imshow(self.name, frame)

    @property
    def name(self) -> str:
        return self._name

class DetectionsVisualizer(BaseVisualizer):
    detectionStream: str # Detection stream name
    labels: List[Union[str, Tuple[str, str]]] = None
    aspectRatio: Tuple[int,int] = None

    def __init__(self,
        frameStream: str,
        detectionsStream: str,
        labels: List[Union[str, Tuple[str, str]]] = None,
        aspectRatio: Tuple[int,int] = None,
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
        if labels is not None:
            self.labels = [label if isinstance(label, str) else label[0] for label in labels]
        self.aspectRatio = aspectRatio
        self.detectionStream = detectionsStream
    
    def newMsgs(self, msgs: Dict):
        frame = msgs[super().name].getCvFrame()
        dets = msgs[self.detectionStream]
        drawDetections(frame, dets, self.labels, self.aspectRatio)
        super().newMsgs(frame=frame)

class Visualizer:
    _components: List[Component]
    _visualizers: List[BaseVisualizer] = []

    def __init__(self, components: List[Component], scale: Union[None, float, Tuple[int,int]] = None,fps=False) -> None:
        self._components = components

        nns = self._components_by_type(components, NNComponent)
        frames = self._streams_by_type(components, dai.ImgFrame)

        for nn in nns:
            for frame in frames:
                nnStreamNames = self._streams_by_type_xout(nn.xouts, dai.ImgDetections)
                detVis = DetectionsVisualizer(frame, nnStreamNames[0], nn.labels, aspectRatio=nn.size if frame != 'passthrough' else None)
                detVis.setBase(scale, fps)
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

    def _streams_by_type_xout(self, xouts:  Dict[str, Tuple[type, type]], type: Type) -> List[str]:
        streams = []
        for name, (compType, daiType) in xouts.items():
            if type == daiType:
                streams.append(name)
        return streams
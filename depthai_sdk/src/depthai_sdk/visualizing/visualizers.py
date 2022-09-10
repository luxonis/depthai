from typing import Dict
import time
from ..components import NNComponent, ComponentGroup
from enum import IntEnum
from .visualizer_helper import *


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


class FPS:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self) -> float:
        diff = self.timestamp - self.start
        return self.frame_cnt / diff if diff != 0 else 0.0


class BaseVisualizer:
    _frame_stream: str
    _scale: Union[None, float, Tuple[int, int]] = None
    _fps: Dict[str, FPS] = None
    _callback: Callable = None

    def __init__(self, frameStream: str) -> None:
        self._frame_stream = frameStream

    def setBase(self,
                scale: Union[None, float, Tuple[int, int]] = None,
                fps: Dict[str, FPS] = None,
                callback: Callable = None
                ):
        self._scale = scale
        self._fps = fps
        self._callback = callback

    def newMsgs(self, input: Union[Dict, dai.ImgFrame, np.ndarray]):
        msgs = input
        if isinstance(input, Dict):
            input = input[self._frame_stream]

        if isinstance(input, dai.ImgFrame):
            frame = input.getCvFrame()
        elif isinstance(input, np.ndarray):
            frame = input
        else:
            ValueError(f'BaseSync received {type(input)} on newMsgs function! It requires Union[Dict, dai.ImgFrame, np.ndarray]')


        if self._fps:
            i = 0
            for name, handler in self._fps.items():
                putText(frame, "{} FPS: {:.1f}".format(name, handler.fps()), (10, 20 + i * 20), scale=0.7)
                i += 1

        if self._scale:
            if isinstance(self._scale, Tuple):
                frame = cv2.resize(frame, self._scale)  # Resize frame
            elif isinstance(self._scale, float):
                frame = cv2.resize(frame, (
                    int(frame.shape[1] * self._scale),
                    int(frame.shape[0] * self._scale)
                ))

        if self._callback:  # Don't display frame, call the callback
            self._callback(msgs, frame)
        else:
            cv2.imshow(self.name, frame)

    @property
    def name(self) -> str:
        return self._frame_stream

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

        putText(frame, text, (x, y))


class DetectionVisualizer(BaseVisualizer):
    detectionStream: str  # Detection stream name
    labels: List[Tuple[str, Tuple[int, int, int]]] = None
    normalizer: NormalizeBoundingBox

    def __init__(self,
                 frameStream: str,
                 detectionsStream: str,
                 detectorComp: NNComponent
                 ) -> None:
        """
        Visualizes object detection results.

        Args:
            frameStream (str): Name of the frame stream to which we will draw detection results
            detectionsStream (str): Name of the detections stream
        """
        super().__init__(frameStream)
        # TODO: add support for colors, generate new colors for each label that doesn't have colors
        if detectorComp.labels:
            self.labels = []
            n_colors = [isinstance(label, str) for label in detectorComp.labels].count(True)
            # np.array of (b,g,r), 0..1
            colors = np.array(distinctipy.get_colors(n_colors=n_colors, rng=123123, pastel_factor=0.5))[..., ::-1]
            colors = [distinctipy.get_rgb256(clr) for clr in colors]  # List of (b,g,r), 0..255
            for label in detectorComp.labels:
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
        self.normalizer = NormalizeBoundingBox(detectorComp.size, detectorComp.arResizeMode)

    def get_imgFrame(self, msgs: Dict) -> dai.ImgFrame:
        return msgs[super().name]

    def get_imgDetections(self, msgs: Dict) -> dai.ImgDetections:
        return msgs[self.detectionStream]

    def newMsgs(self, msgs: Dict):
        imgFrame = self.get_imgFrame(msgs)
        frame = imgFrame.getCvFrame()
        dets = self.get_imgDetections(msgs)
        drawDetections(frame, dets, self.normalizer, self.labels)
        msgs[super().name] = frame
        super().newMsgs(msgs)


class DetectionClassificationVisualizer(DetectionVisualizer):

    def __init__(self,
                 frame_name: str,
                 group: ComponentGroup
                 ) -> None:

        super().__init__(frameStream=frame_name,
                         detectionsStream=group.nn_name,
                         detectorComp=group.nn_component)

        self.classificationStream = group.second_nn_name

    def newMsgs(self, msgs: Dict):
        # print('2nd stage results:',msgs[self.classificationStream])
        super().newMsgs(msgs)

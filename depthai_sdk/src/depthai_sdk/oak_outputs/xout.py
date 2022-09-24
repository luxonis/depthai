import cv2
import depthai as dai
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

import numpy as np
from distinctipy import distinctipy

from .xout_base import XoutBase, StreamXout
from ..classes.packets import FramePacket, SpatialBbMappingPacket, DetectionPacket, TwoStagePacket
from .visualizer_helper import Visualizer, FPS, colorizeDisparity, calc_disp_multiplier, drawMappings, drawDetections, \
    hex_to_bgr
from .normalize_bb import NormalizeBoundingBox

"""
Xout classes are abstracting streaming messages to the host computer (via XLinkOut) and syncing those messages
on the host side before sending (synced) messages to message sinks (eg. visualizers, or loggers).
TODO:
- separate syncing logic from the class. XoutTwoStage should extend the XoutNnResults (currently can't as syncing logic is not separated)
"""

class XoutFrames(XoutBase):
    """
    Single message, no syncing required
    """
    frames: StreamXout
    _scale: Union[None, float, Tuple[int, int]] = None
    _fps: Dict[str, FPS] = None

    def __init__(self, frames: StreamXout):
        self.frames = frames
        super().__init__()

    def setup_visualize(self,
                scale: Union[None, float, Tuple[int, int]] = None,
                fps: Dict[str, FPS] = None,
                ):
        self._scale = scale
        self._fps = fps
        self._vis = True

    def visualize(self, packet: FramePacket) -> None:
        """
        Called from main thread if vis=True
        """

        if self._fps:
            i = 0
            for name, handler in self._fps.items():
                Visualizer.putText(packet.frame, "{} FPS: {:.1f}".format(name, handler.fps()), (10, 20 + i * 20),
                                   scale=0.7)
                i += 1

        if self._scale:
            if isinstance(self._scale, Tuple):
                packet.frame = cv2.resize(packet.frame, self._scale)  # Resize frame
            elif isinstance(self._scale, float):
                packet.frame = cv2.resize(packet.frame, (
                    int(packet.frame.shape[1] * self._scale),
                    int(packet.frame.shape[0] * self._scale)
                ))

        if self.callback:  # Don't display frame, call the callback
            self.callback(packet)
        else:
            cv2.imshow(packet.name, packet.frame)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames]

    def newMsg(self, name: str, msg) -> None:
        if name not in self._streams: return

        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        packet = FramePacket(name, msg, msg.getCvFrame())

        self.queue.put(packet, block=False)

class XoutDisparity(XoutFrames):
    multiplier: float
    def __init__(self, frames: StreamXout, max_disp: float):
        super().__init__(frames)
        self.multiplier = 255.0 / max_disp

    def visualize(self, packet: FramePacket):
        packet.frame = colorizeDisparity(packet.imgFrame, self.multiplier)
        super().visualize(packet)

class XoutDepth(XoutFrames):
    factor: float = None
    def __init__(self, device: dai.Device, frames: StreamXout):
        super().__init__(frames)
        self.device = device
        self.multiplier = 255 / 95.0

    def visualize(self, packet: FramePacket):
        if not self.factor:
            size = (packet.imgFrame.getWidth(), packet.imgFrame.getHeight())
            self.factor = calc_disp_multiplier(self.device, size)

        depth = np.array(packet.imgFrame.getFrame())
        with np.errstate(divide='ignore'):
            disp = (self.factor / depth).astype(np.uint8)

        packet.frame = colorizeDisparity(disp, multiplier=self.multiplier)
        super().visualize(packet)

class XoutSpatialBbMappings(XoutFrames):
    # Streams
    frames: StreamXout
    configs: StreamXout

    # Save messages
    depth_msg: Optional[dai.ImgFrame] = None
    config_msg: Optional[dai.SpatialLocationCalculatorConfig] = None

    factor: float = None

    def __init__(self, device: dai.Device, frames: StreamXout, configs: StreamXout):
        self.frames = frames
        self.configs = configs
        self.device = device
        self.multiplier = 255 / 95.0
        super().__init__(frames)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.configs]

    def visualize(self, packet: SpatialBbMappingPacket):
        if not self.factor:
            size = (packet.imgFrame.getWidth(), packet.imgFrame.getHeight())
            self.factor = calc_disp_multiplier(self.device, size)

        depth = np.array(packet.imgFrame.getFrame())
        with np.errstate(divide='ignore'):
            disp = (self.factor / depth).astype(np.uint8)

        packet.frame = colorizeDisparity(disp, multiplier=self.multiplier)
        drawMappings(packet)

        super().visualize(packet)

    def newMsg(self, name: str, msg: dai.Buffer) -> None:
        # Ignore frames that we aren't listening for
        if name not in self._streams: return

        if name == self.frames.name:
            self.depth_msg = msg

        if name == self.configs.name:
            self.config_msg = msg

        if self.depth_msg and self.config_msg:

            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            packet = SpatialBbMappingPacket(
                self.frames.name,
                self.depth_msg,
                self.config_msg
            )

            self.queue.put(packet, block=False)

            self.config_msg = None
            self.depth_msg = None


class XoutNnResults(XoutFrames):
    nn_results: StreamXout

    labels: List[Tuple[str, Tuple[int, int, int]]] = None
    normalizer: NormalizeBoundingBox

    msgs: Dict[str, Dict[str, dai.Buffer]]  # List of messages.
    """
    msgs = {seq: {stream_name: frame}}
    Example:

    msgs = {
        '1': {
            'rgb': dai.Frame(),
            'dets': dai.ImgDetections(),
        }
        '2': {
            'rgb': dai.Frame(),
            'dets': dai.ImgDetections(),
        }
    }
    """

    def __init__(self, detNn, frames: StreamXout, nn_results: StreamXout):
        self.nn_results = nn_results
        # Save StreamXout before initializing super()!
        super().__init__(frames)
        self.detNn = detNn
        self.msgs = dict()

        # TODO: add support for colors, generate new colors for each label that doesn't have colors
        if detNn.labels:
            self.labels = []
            n_colors = [isinstance(label, str) for label in detNn.labels].count(True)
            # np.array of (b,g,r), 0..1
            colors = np.array(distinctipy.get_colors(n_colors=n_colors, rng=123123, pastel_factor=0.5))[..., ::-1]
            colors = [distinctipy.get_rgb256(clr) for clr in colors]  # List of (b,g,r), 0..255
            for label in detNn.labels:
                if isinstance(label, str):
                    text = label
                    color = colors.pop(0)  # Take last row
                elif isinstance(label, list):
                    text = label[0]
                    color = hex_to_bgr(label[1])
                else:
                    raise ValueError('Model JSON config error. Label map list can have either str or list!')

                self.labels.append((text, color))

        self.normalizer = NormalizeBoundingBox(detNn.size, detNn.arResizeMode)


    def visualize(self, packet: DetectionPacket):
        drawDetections(packet, self.normalizer, self.labels)
        super().visualize(packet)

    def newMsg(self, name: str, msg) -> None:
        # Ignore frames that we aren't listening for
        if name not in self._streams: return

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs: self.msgs[seq] = dict()
        self.msgs[seq][name] = msg

        if len(self._streams) == len(self.msgs[seq]):  # We have sequence num synced frames!

            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            packet = DetectionPacket(
                self.frames.name,
                self.msgs[seq][self.frames.name],
                self.msgs[seq][self.nn_results.name],
            )

            self.queue.put(packet, block=False)

            # Remove previous msgs (memory cleaning)
            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results]


# class TimestampSycn(BaseSync):
#     """
#     Timestamp sync will sync all streams based on the timestamp
#     """
#     msgs: Dict[str, List[dai.Buffer]] = dict()  # List of messages
#
#     def newMsg(self, name: str, msg) -> None:
#         # Ignore frames that we aren't listening for
#         if name not in self.streams: return
#         # Return all latest msgs (not synced)
#         if name not in self.msgs: self.msgs[name] = []
#
#         self.msgs[name].append(msg)
#         msgsAvailableCnt = [0 < len(arr) for _, arr in self.msgs.items()].count(True)
#
#         if len(self.streams) == msgsAvailableCnt:
#             # We have at least 1 msg for each stream. Get the latest, remove all others.
#             ret = {}
#             for name, arr in self.msgs.items():
#                 # print(f'len(msgs[{name}])', len(self.msgs[name]))
#                 self.msgs[name] = self.msgs[name][-1:]  # Remove older msgs
#                 # print(f'After removing - len(msgs[{name}])', len(self.msgs[name]))
#                 ret[name] = arr[-1]
#
#             if self.queue.full():
#                 self.queue.get()  # Get one, so queue isn't full
#
#             # print(time.time(),' Putting msg batch into queue. queue size', self.queue.qsize(), 'self.msgs len')
#
#             self.queue.put(ret, block=False)


class XoutTwoStage(XoutNnResults):
    """
    Two stage syncing based on sequence number. Each frame produces ImgDetections msg that contains X detections.
    Each detection (if not on blacklist) will crop the original frame and forward it to the second (stage) NN for
    inferencing.
    """
    msgs: Dict[str, Dict[str, Any]] = dict()  # List of messages
    """
    msgs = {
        '1': TwoStageSyncPacket(),
        '2': TwoStageSyncPacket(), 
    }
    """
    whitelist_labels: Optional[List[int]] = None
    scaleBb: Optional[Tuple[int, int]] = None

    second_nn: StreamXout

    def __init__(self, detNn, secondNn, frames: StreamXout, detections: StreamXout, second_nn: StreamXout):
        self.second_nn = second_nn
        # Save StreamXout before initializing super()!
        super().__init__(detNn, frames, detections)

        self.detNn = detNn
        self.secondNn = secondNn

        conf = detNn._multi_stage_config # No types due to circular import...
        if conf is not None:
            self.labels = conf.labels
            self.scaleBb = conf.scaleBb

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results, self.second_nn]

    # No need for `def visualize()` as `XoutNnResults.visualize()` does what we want

    def newMsg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams: return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()
            self.msgs[seq][self.second_nn.name] = []
            self.msgs[seq][self.nn_results.name] = None

        if name == self.second_nn.name:
            self.msgs[seq][name].append(msg)
            # print(f'Added recognition seq {seq}, total len {len(self.msgs[seq]["recognition"])}')
        elif name == self.nn_results.name:
            self.add_detections(seq, msg)
            # print(f'Added detection seq {seq}')
        elif name in self.frames.name:
            self.msgs[seq][name] = msg
            # print(f'Added frame seq {seq}')
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if self.synced(seq):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            packet = TwoStagePacket(
                self.frames.name,
                self.msgs[seq][self.frames.name],
                self.msgs[seq][self.nn_results.name],
                self.msgs[seq][self.second_nn.name],
                self.whitelist_labels
            )
            self.queue.put(packet, block=False)

            # Throws RuntimeError: dictionary changed size during iteration
            # for s in self.msgs:
            #     if int(s) <= int(seq):
            #         del self.msgs[s]

            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs

    def add_detections(self, seq: str, dets: dai.ImgDetections):
        # Used to match the scaled bounding boxes by the 2-stage NN script node
        self.msgs[seq][self.nn_results.name] = dets

        if self.scaleBb is None: return  # No scaling required, ignore

        for det in dets.detections:
            # Skip resizing BBs if we have whitelist and the detection label is not on it
            if self.labels and det.label not in self.labels: continue
            det.xmin -= self.scaleBb[0] / 100
            det.ymin -= self.scaleBb[1] / 100
            det.xmax += self.scaleBb[0] / 100
            det.ymax += self.scaleBb[1] / 100

    def synced(self, seq: str) -> bool:
        """
        Messages are in sync if:
            - dets is not None
            - We have at least one ImgFrame
            - number of recognition msgs is sufficient
        """
        packet = self.msgs[seq]

        if self.frames.name not in packet:
            return False  # We don't have required ImgFrames

        if not packet[self.nn_results.name]:
            return False  # We don't have dai.ImgDetections

        if len(packet[self.second_nn.name]) < self.required_recognitions(seq):
            return False  # We don't have enough 2nd stage NN results

        return True

    def required_recognitions(self, seq: str) -> int:
        """
        Required recognition results for this packet, which depends on number of detections (and white-list labels)
        """
        dets: List[dai.ImgDetection] = self.msgs[seq][self.nn_results.name].detections
        if self.whitelist_labels:
            return len([det for det in dets if det.label in self.whitelist_labels])
        else:
            return len(dets)

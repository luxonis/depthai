import threading
from typing import List, Union, Dict, Any, Optional, Tuple

import depthai as dai

from depthai_sdk.classes import Detections, ImgLandmarks, SemanticSegmentation
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.classes.packets import (
    Detection,
    DetectionPacket,
    ImgLandmarksPacket,
    NnOutputPacket,
    SemanticSegmentationPacket,
    SpatialBbMappingPacket,
    TwoStagePacket,
    NNDataPacket
)
from depthai_sdk.oak_outputs.syncing import SequenceNumSync
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from depthai_sdk.oak_outputs.xout.xout_depth import XoutDisparityDepth
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_seq_sync import XoutSeqSync
from depthai_sdk.types import XoutNNOutputPacket
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.colors import generate_colors, hex_to_bgr


class XoutNnData(XoutBase):
    def __init__(self, xout: StreamXout):
        self.nndata_out = xout
        super().__init__()

    def xstreams(self) -> List[StreamXout]:
        return [self.nndata_out]

    def new_msg(self, name: str, msg: dai.NNData) -> NNDataPacket:
        if name not in self._streams:
            return
        return NNDataPacket(name=self.get_packet_name(), nn_data=msg)


class XoutNnResults(XoutSeqSync, XoutFrames):
    def xstreams(self) -> List[StreamXout]:
        return [self.nn_results, self.frames]

    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout,
                 bbox: BoundingBox):
        self.det_nn = det_nn
        self.nn_results = nn_results

        XoutFrames.__init__(self, frames)
        XoutSeqSync.__init__(self, [frames, nn_results])

        self.name = 'NN results'
        self.labels = None
        self.bbox = bbox

        # TODO: add support for colors, generate new colors for each label that doesn't have colors
        if det_nn._labels:
            self.labels = []
            n_colors = [isinstance(label, str) for label in det_nn._labels].count(True)
            # List of colors in BGR format
            colors = generate_colors(n_colors)

            for label in det_nn._labels:
                if isinstance(label, str):
                    text = label
                    color = colors.pop(0)  # Take last row
                elif isinstance(label, list):
                    text = label[0]
                    color = hex_to_bgr(label[1])
                else:
                    raise ValueError('Model JSON config error. Label map list can have either str or list!')

                self.labels.append((text, color))

        self._resize_mode: ResizeMode = det_nn._ar_resize_mode
        self._nn_size: Tuple[int, int] = det_nn._size

    def package(self, msgs: Dict) -> XoutNNOutputPacket:
        nn_result = msgs[self.nn_results.name]
        img = msgs[self.frames.name]
        if type(nn_result) == dai.NNData:
            decode_fn = self.det_nn._decode_fn

            if decode_fn is None:
                return NnOutputPacket(self.get_packet_name(), img, nn_result, self.bbox)

            decoded_nn_result = decode_fn(nn_result)
            if type(decoded_nn_result) == Detections:
                packet = DetectionPacket(self.get_packet_name(), img, nn_result, self.bbox)
                return self._add_detections_to_packet(packet, decoded_nn_result)
            elif type(decoded_nn_result) == ImgLandmarks:
                return ImgLandmarksPacket(self.get_packet_name(), img, nn_result, decoded_nn_result, self.bbox)
            elif type(decoded_nn_result) == SemanticSegmentation:
                return SemanticSegmentationPacket(self.get_packet_name(), img, nn_result, decoded_nn_result, self.bbox)
            raise ValueError(f'NN result decoding failed! decode() returned type {type(nn_result)}')

        elif type(nn_result) in [dai.ImgDetections, dai.SpatialImgDetections]:
            packet = DetectionPacket(self.get_packet_name(), img, nn_result, self.bbox)
            return self._add_detections_to_packet(packet, nn_result)
        else:
            raise ValueError(f'Unknown NN result type: {type(nn_result)}')

    def _add_detections_to_packet(self,
                                  packet: DetectionPacket,
                                  dets: Union[dai.ImgDetections, dai.SpatialImgDetections, Detections]
                                  ) -> DetectionPacket:
        for detection in dets.detections:
            packet.detections.append(Detection(
                img_detection=detection if isinstance(detection, dai.ImgDetection) else None,
                label_str=self.labels[detection.label][0] if self.labels else str(detection.label),
                confidence=detection.confidence,
                color=self.labels[detection.label][1] if self.labels else (255, 255, 255),
                bbox=BoundingBox(detection),
                angle=detection.angle if hasattr(detection, 'angle') else None,
                ts=dets.getTimestamp()
            ))
        return packet


class XoutSpatialBbMappings(XoutDisparityDepth, SequenceNumSync):
    def __init__(self,
                 device: dai.Device,
                 stereo: dai.node.StereoDepth,
                 frames: StreamXout,  # passthroughDepth
                 configs: StreamXout,  # out
                 dispScaleFactor: float,
                 bbox: BoundingBox):
        self._stereo = stereo
        self.frames = frames
        self.configs = configs
        self.bbox = bbox

        XoutDisparityDepth.__init__(self, device, frames, dispScaleFactor, None)
        SequenceNumSync.__init__(self, 2)

    def new_msg(self, name: str, msg):
        # Ignore frames that we aren't listening for
        if name not in self._streams: return

        synced = self.sync(msg.getSequenceNum(), name, msg)
        if synced:
            return self.package(synced)

    def on_callback(self, packet) -> None:
        pass

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.configs]

    def package(self, msgs: Dict) -> SpatialBbMappingPacket:
        return SpatialBbMappingPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            msgs[self.configs.name],
            disp_scale_factor=self.disp_scale_factor,
        )


class XoutTwoStage(XoutNnResults):
    """
    Two stage syncing based on sequence number. Each frame produces ImgDetections msg that contains X detections.
    Each detection (if not on blacklist) will crop the original frame and forward it to the second (stage) NN for
    inferencing.

    msgs = {
        '1': TwoStageSyncPacket(),
        '2': TwoStageSyncPacket(),
    }
    """

    def __init__(self,
                 det_nn: 'NNComponent',
                 second_nn: 'NNComponent',
                 frames: StreamXout,
                 det_out: StreamXout,
                 second_nn_out: StreamXout,
                 device: dai.Device,
                 input_queue_name: str,
                 bbox: BoundingBox):
        self.second_nn_out = second_nn_out
        # Save StreamXout before initializing super()!
        super().__init__(det_nn, frames, det_out, bbox)

        self.msgs: Dict[str, Dict[str, Any]] = dict()
        self.det_nn = det_nn
        self.second_nn = second_nn
        self.name = 'Two-stage detection'

        self.whitelist_labels: Optional[List[int]] = second_nn._multi_stage_nn.whitelist_labels
        self.scale_bb: Optional[Tuple[int, int]] = second_nn._multi_stage_nn.scale_bb

        self.device = device
        self.input_queue_name = input_queue_name
        self.input_queue = None
        self.input_cfg_queue = None

        self.lock = threading.Lock()

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results, self.second_nn_out]

    # No need for `def visualize()` as `XoutNnResults.visualize()` does what we want

    def new_msg(self, name: str, msg: dai.Buffer):
        if name not in self._streams:
            return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()
            self.msgs[seq][self.second_nn_out.name] = []
            self.msgs[seq][self.nn_results.name] = None

        if name == self.second_nn_out.name:
            fn = self.second_nn._decode_fn
            if fn is not None:
                self.msgs[seq][name].append(fn(msg))
            else:
                self.msgs[seq][name].append(msg)

        elif name == self.nn_results.name:
            fn = self.det_nn._decode_fn
            if fn is not None:
                msg = fn(msg)

            self.add_detections(seq, msg)

            if self.input_queue_name:
                # We cannot create them in __init__ as device is not initialized yet
                if self.input_queue is None:
                    self.input_queue = self.device.getInputQueue(self.input_queue_name,
                                                                 maxSize=8,
                                                                 blocking=False)
                    self.input_cfg_queue = self.device.getInputQueue(self.input_queue_name + '_cfg',
                                                                     maxSize=8,
                                                                     blocking=False)

                for i, det in enumerate(msg.detections):
                    cfg = dai.ImageManipConfig()

                    if isinstance(det, dai.ImgDetection):
                        rect = (det.xmin, det.ymin, det.xmax, det.ymax)
                    else:
                        rect = det[0], det[1], det[2], det[3]

                    try:
                        angle = msg.angles[i]
                        if angle != 0.0:
                            rr = dai.RotatedRect()
                            rr.center.x = rect[0]
                            rr.center.y = rect[1]
                            rr.size.width = rect[2] - rect[0]
                            rr.size.height = rect[3] - rect[1]
                            rr.angle = angle
                            cfg.setCropRotatedRect(rr, normalizedCoords=True)
                        else:
                            cfg.setCropRect(rect)
                    except AttributeError:
                        cfg.setCropRect(rect)

                    cfg.setResize(*self.second_nn._size)

                    if i == 0:
                        try:
                            frame = self.msgs[seq][self.frames.name]
                        except KeyError:
                            continue

                        self.input_queue.send(frame)
                        cfg.setReusePreviousImage(True)

                    self.input_cfg_queue.send(cfg)

        elif name in self.frames.name:
            self.msgs[seq][name] = msg
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if self.synced(seq):
            # Frames synced!
            dets = self.msgs[seq][self.nn_results.name]
            packet = TwoStagePacket(
                self.get_packet_name(),
                self.msgs[seq][self.frames.name],
                dets,
                self.msgs[seq][self.second_nn_out.name],
                self.whitelist_labels,
                self.bbox
            )

            with self.lock:
                new_msgs = {}
                for name, msg in self.msgs.items():
                    if int(name) > int(seq):
                        new_msgs[name] = msg
                self.msgs = new_msgs

            return self._add_detections_to_packet(packet, dets)

    def add_detections(self, seq: str, dets: dai.ImgDetections):
        # Used to match the scaled bounding boxes by the 2-stage NN script node
        self.msgs[seq][self.nn_results.name] = dets

        if isinstance(dets, dai.ImgDetections):
            if self.scale_bb is None:
                return  # No scaling required, ignore

            for det in dets.detections:
                # Skip resizing BBs if we have whitelist and the detection label is not on it
                if self.labels and det.label not in self.labels: continue
                det.xmin -= self.scale_bb[0] / 100
                det.ymin -= self.scale_bb[1] / 100
                det.xmax += self.scale_bb[0] / 100
                det.ymax += self.scale_bb[1] / 100

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

        if len(packet[self.second_nn_out.name]) < self.required_recognitions(seq):
            return False  # We don't have enough 2nd stage NN results

        # print('Synced!')
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

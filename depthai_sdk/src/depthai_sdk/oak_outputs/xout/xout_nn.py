import threading
from typing import List, Union, Dict, Any, Optional, Tuple

import depthai as dai
import numpy as np

from depthai_sdk.classes import Detections, ImgLandmarks, SemanticSegmentation
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.classes.packets import (
    _Detection, DetectionPacket, TrackerPacket, SpatialBbMappingPacket, TwoStagePacket, NNDataPacket
)
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_seq_sync import XoutSeqSync
from depthai_sdk.visualize.visualizer import Visualizer
from depthai_sdk.visualize.visualizer_helper import hex_to_bgr, colorize_disparity, draw_mappings, depth_to_disp_factor
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.colors import generate_colors
try:
    import cv2
except ImportError:
    cv2 = None

class XoutNnData(XoutBase):
    def __init__(self, xout: StreamXout):
        self.nndata_out = xout
        super().__init__()
        self.name = 'NNData'

    def visualize(self, packet: NNDataPacket):
        print('Visualization of NNData is not supported')

    def xstreams(self) -> List[StreamXout]:
        return [self.nndata_out]

    def new_msg(self, name: str, msg: dai.NNData) -> None:
        if name not in self._streams:
            return

        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        packet = NNDataPacket(name=self.name, nn_data=msg)
        self.queue.put(packet, block=False)


class XoutNnResults(XoutSeqSync, XoutFrames):
    def xstreams(self) -> List[StreamXout]:
        return [self.nn_results, self.frames]

    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout):
        self.det_nn = det_nn
        self.nn_results = nn_results

        XoutFrames.__init__(self, frames)
        XoutSeqSync.__init__(self, [frames, nn_results])

        self.name = 'NN results'
        self.labels = None

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

        self.segmentation_colormap = None

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        super().setup_visualize(visualizer, visualizer_enabled, name)

    def on_callback(self, packet: Union[DetectionPacket, TrackerPacket]):
        # Convert Grayscale to BGR
        if len(packet.frame.shape) == 2:
            packet.frame = np.dstack((packet.frame, packet.frame, packet.frame))

        frame_shape = self.det_nn._input.stream_size[::-1]

        if self._frame_shape is None:
            # Lazy-load the frame shape
            self._frame_shape = np.array([*frame_shape])
            if self._visualizer:
                self._visualizer.frame_shape = self._frame_shape

        bbox = BoundingBox().resize_to_aspect_ratio(self._frame_shape, self._nn_size, self._resize_mode)

        # Add detections to packet
        if isinstance(packet.img_detections, dai.ImgDetections) \
                or isinstance(packet.img_detections, dai.SpatialImgDetections) \
                or isinstance(packet.img_detections, Detections):

            for detection in packet.img_detections.detections:
                d = _Detection()
                d.img_detection = detection
                d.label = self.labels[detection.label][0] if self.labels else str(detection.label)
                d.color = self.labels[detection.label][1] if self.labels else (255, 255, 255)

                d.top_left, d.bottom_right = bbox.get_relative_bbox(BoundingBox(detection)).denormalize(self._frame_shape)
                packet.detections.append(d)

            if self._visualizer:
                # Add detections to visualizer
                self._visualizer.add_detections(
                    packet.img_detections.detections,
                    bbox,
                    self.labels,
                    is_spatial=packet._is_spatial_detection()
                )
        elif isinstance(packet.img_detections, ImgLandmarks):
            if not self._visualizer:
                return

            all_landmarks = packet.img_detections.landmarks
            all_landmarks_indices = packet.img_detections.landmarks_indices
            colors = packet.img_detections.colors
            for landmarks, indices in zip(all_landmarks, all_landmarks_indices):
                for i, landmark in enumerate(landmarks):
                    # Map normalized coordinates to frame coordinates
                    l = [(int(point[0] * self._frame_shape[1]), int(point[1] * self._frame_shape[0])) for point in landmark]
                    idx = indices[i]

                    self._visualizer.add_line(pt1=tuple(l[0]), pt2=tuple(l[1]), color=colors[idx], thickness=4)
                    self._visualizer.add_circle(coords=tuple(l[0]), radius=8, color=colors[idx], thickness=-1)
                    self._visualizer.add_circle(coords=tuple(l[1]), radius=8, color=colors[idx], thickness=-1)
        elif isinstance(packet.img_detections, SemanticSegmentation):
            raise NotImplementedError('Semantic segmentation visualization is not implemented yet!')
            if not self._visualizer:
                return

            # Generate colormap if not already generated
            if self.segmentation_colormap is None:
                n_classes = len(self.labels) if self.labels else 8
                self.segmentation_colormap = generate_colors(n_classes)

            mask = np.array(packet.img_detections.mask).astype(np.uint8)

            if mask.ndim == 3:
                mask = np.argmax(mask, axis=0)

            try:
                colorized_mask = np.array(self.segmentation_colormap)[mask]
            except IndexError:
                unique_classes = np.unique(mask)
                max_class = np.max(unique_classes)
                new_colors = generate_colors(max_class - len(self.segmentation_colormap) + 1)
                self.segmentation_colormap.extend(new_colors)
                colorized_mask = np.array(self.segmentation_colormap)[mask]

            # bbox = None
            # if self.normalizer.resize_mode == ResizeMode.LETTERBOX:
            #     bbox = self.normalizer.get_letterbox_bbox(packet.frame, normalize=True)
            #     input_h, input_w = self.normalizer.aspect_ratio
            #     resize_bbox = bbox[0] * input_w, bbox[1] * input_h, bbox[2] * input_w, bbox[3] * input_h
            #     resize_bbox = np.int0(resize_bbox)
            # else:
            #     resize_bbox = self.normalizer.normalize(frame=np.zeros(self._frame_shape, dtype=bool),
            #                                             bbox=bbox or (0., 0., 1., 1.))

            # x1, y1, x2, y2 = resize_bbox
            # h, w = packet.frame.shape[:2]
            # # Stretch mode
            # if self.normalizer.resize_mode == ResizeMode.STRETCH:
            #     colorized_mask = cv2.resize(colorized_mask, (w, h))
            # elif self.normalizer.resize_mode == ResizeMode.LETTERBOX:
            #     colorized_mask = cv2.resize(colorized_mask[y1:y2, x1:x2], (w, h))
            # else:
            #     padded_mask = np.zeros((h, w, 3), dtype=np.uint8)
            #     resized_mask = cv2.resize(colorized_mask, (x2 - x1, y2 - y1))
            #     padded_mask[y1:y2, x1:x2] = resized_mask
            #     colorized_mask = padded_mask

            # self._visualizer.add_mask(colorized_mask, alpha=0.5)

    def package(self, msgs: Dict):
        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        decode_fn = self.det_nn._decode_fn
        packet = DetectionPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            msgs[self.nn_results.name] if decode_fn is None else decode_fn(msgs[self.nn_results.name]),
            self._visualizer
        )

        self.queue.put(packet, block=False)

class XoutSpatialBbMappings(XoutSeqSync, XoutFrames):
    def __init__(self,
                 device: dai.Device,
                 stereo: dai.node.StereoDepth,
                 frames: StreamXout,
                 configs: StreamXout):
        self._stereo = stereo
        self.frames = frames
        self.configs = configs

        XoutFrames.__init__(self, frames)
        XoutSeqSync.__init__(self, [frames, configs])

        self.device = device
        self.multiplier = 255 / 95.0
        self.factor = None
        self.name = 'Depth & Bounding Boxes'

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.configs]

    def visualize(self, packet: SpatialBbMappingPacket):
        if not self.factor:
            size = (packet.msg.getWidth(), packet.msg.getHeight())
            self.factor = depth_to_disp_factor(self.device, self._stereo)

        depth = np.array(packet.msg.getFrame())
        with np.errstate(all='ignore'):
            disp = (self.factor / depth).astype(np.uint8)

        print('disp max', np.max(disp), 'disp min', np.min(disp))
        packet.frame = colorize_disparity(disp, multiplier=1)
        draw_mappings(packet)

        super().visualize(packet)

    def package(self, msgs: Dict):
        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full
        packet = SpatialBbMappingPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            msgs[self.configs.name],
            self._visualizer
        )
        self.queue.put(packet, block=False)


class XoutTwoStage(XoutNnResults):
    """
    Two stage syncing based on sequence number. Each frame produces ImgDetections msg that contains X detections.
    Each detection (if not on blacklist) will crop the original frame and forward it to the second (stage) NN for
    inferencing.
    """
    """
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
                 input_queue_name: str):
        self.second_nn_out = second_nn_out
        # Save StreamXout before initializing super()!
        super().__init__(det_nn, frames, det_out)

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

    def new_msg(self, name: str, msg: dai.Buffer) -> None:
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

            # print(f'Added detection seq {seq}')
        elif name in self.frames.name:
            self.msgs[seq][name] = msg
            # print(f'Added frame seq {seq}')
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if self.synced(seq):
            # print('Synced', seq)
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            packet = TwoStagePacket(
                self.get_packet_name(),
                self.msgs[seq][self.frames.name],
                self.msgs[seq][self.nn_results.name],
                self.msgs[seq][self.second_nn_out.name],
                self.whitelist_labels,
                self._visualizer
            )
            self.queue.put(packet, block=False)

            # Throws RuntimeError: dictionary changed size during iteration
            # for s in self.msgs:
            #     if int(s) <= int(seq):
            #         del self.msgs[s]

            with self.lock:
                new_msgs = {}
                for name, msg in self.msgs.items():
                    if int(name) > int(seq):
                        new_msgs[name] = msg
                self.msgs = new_msgs

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

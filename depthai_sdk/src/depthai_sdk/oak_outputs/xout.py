from abc import abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union

import cv2
import depthai as dai
import numpy as np
from distinctipy import distinctipy

from depthai_sdk.classes.nn_results import Detections, ImgLandmarks, SemanticSegmentation
from depthai_sdk.classes.packets import (
    FramePacket,
    SpatialBbMappingPacket,
    DetectionPacket,
    TwoStagePacket,
    TrackerPacket,
    IMUPacket, DepthPacket, _Detection
)
from depthai_sdk.components.nn_helper import ResizeMode
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox
from depthai_sdk.oak_outputs.syncing import SequenceNumSync
from depthai_sdk.oak_outputs.xout_base import XoutBase, StreamXout
from depthai_sdk.recorders.video_recorder import VideoRecorder
from depthai_sdk.recorders.video_writers import AvWriter
from depthai_sdk.visualize import Visualizer
from depthai_sdk.visualize.configs import StereoColor
from depthai_sdk.visualize.configs import TextPosition
from depthai_sdk.visualize.visualizer import Platform
from depthai_sdk.visualize.visualizer_helper import colorize_disparity, calc_disp_multiplier, draw_mappings, hex_to_bgr

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

    def __init__(self, frames: StreamXout, fps: float = 30, frame_shape: Tuple[int, ...] = None):
        self.frames = frames
        self.name = frames.name

        self.fps = fps
        self._video_recorder = None
        self._is_recorder_enabled = None
        self._frame_shape = frame_shape

        super().__init__()

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        self._visualizer = visualizer
        self._visualizer_enabled = visualizer_enabled
        self.name = name or self.name

    def setup_recorder(self,
                       recorder: VideoRecorder,
                       encoding: str = 'mp4v'):
        self._video_recorder = recorder
        # Enable encoding for the video recorder
        self._video_recorder[self.name].set_fourcc(encoding)

    def visualize(self, packet: FramePacket) -> None:
        """
        Called from main thread if visualizer is not None.
        """

        # Frame shape may be 1D, that means it's an encoded frame
        if self._visualizer.frame_shape is None or np.array(self._visualizer.frame_shape).ndim == 1:
            if self._frame_shape is not None:
                self._visualizer.frame_shape = self._frame_shape
            else:
                self._visualizer.frame_shape = packet.frame.shape

        if self._visualizer.config.output.show_fps:
            self._visualizer.add_text(
                text=f'FPS: {self._fps.fps():.1f}',
                position=TextPosition.TOP_LEFT
            )

        if self.callback:  # Don't display frame, call the callback
            self.callback(packet)
        else:
            packet.frame = self._visualizer.draw(packet.frame)
            # Draw on the frame
            if self._visualizer.platform == Platform.PC:
                cv2.imshow(self.name, packet.frame)
            else:
                pass

    def on_record(self, packet) -> None:
        if self._video_recorder:
            # TODO not ideal to check it this way
            if isinstance(self._video_recorder[self.name], AvWriter):
                self._video_recorder.write(self.name, packet.imgFrame)
            else:
                self._video_recorder.write(self.name, packet.frame)
        # else:
        #     self._video_recorder.add_to_buffer(self.name, packet.frame)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames]

    def new_msg(self, name: str, msg) -> None:
        if name not in self._streams:
            return

        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        packet = FramePacket(name, msg, msg.getCvFrame(), self._visualizer)

        self.queue.put(packet, block=False)

    def __del__(self):
        if self._video_recorder:
            self._video_recorder.close()


class XoutMjpeg(XoutFrames):
    name: str = "MJPEG Stream"

    def __init__(self, frames: StreamXout, color: bool, lossless: bool, fps: float, frame_shape: Tuple[int, ...]):
        super().__init__(frames)
        # We could use cv2.IMREAD_UNCHANGED, but it produces 3 planes (RGB) for mono frame instead of a single plane
        self.flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        self.lossless = lossless
        self.fps = fps
        self._frame_shape = frame_shape

        if lossless and self._visualizer:
            raise ValueError('Visualizing Lossless MJPEG stream is not supported!')

    def decode_frame(self, packet: FramePacket) -> np.ndarray:
        return cv2.imdecode(packet.imgFrame.getData(), self.flag)

    def visualize(self, packet: FramePacket):
        packet.frame = self.decode_frame(packet)
        super().visualize(packet)


class XoutH26x(XoutFrames):
    def __init__(self,
                 frames: StreamXout,
                 color: bool,
                 profile: dai.VideoEncoderProperties.Profile,
                 fps: float,
                 frame_shape: Tuple[int, ...]):
        super().__init__(frames)
        self.name = 'H26x Stream'
        self.color = color
        self.profile = profile
        self.fps = fps
        self._frame_shape = frame_shape
        fourcc = 'hevc' if profile == dai.VideoEncoderProperties.Profile.H265_MAIN else 'h264'

        import av
        self.codec = av.CodecContext.create(fourcc, "r")

    def decode_frame(self, packet: FramePacket):
        enc_packets = self.codec.parse(packet.imgFrame.getData())
        if len(enc_packets) == 0:
            return None

        frames = self.codec.decode(enc_packets[-1])
        if not frames:
            return None

        frame = frames[0].to_ndarray(format='bgr24')

        # If it's Mono, squeeze from 3 planes (height, width, 3) to single plane (height, width)
        if not self.color:
            frame = frame[:, :, 0]

        return frame

    def visualize(self, packet: FramePacket):
        decoded_frame = self.decode_frame(packet)
        if decoded_frame is None:
            return

        packet.frame = decoded_frame
        super().visualize(packet)


class XoutClickable:
    def __init__(self, decay_step: int = 30):
        super().__init__()
        self.buffer = None
        self.decay_step = decay_step

    def on_click_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.buffer = ([0, param[0][y, x], [x, y]])


class XoutDisparity(XoutFrames, XoutClickable):
    name: str = "Disparity"

    def __init__(self,
                 disparity_frames: StreamXout,
                 mono_frames: StreamXout,
                 max_disp: float,
                 fps: float,
                 colorize: StereoColor = None,
                 colormap: int = None,
                 use_wls_filter: bool = None,
                 wls_level: 'WLSLevel' = None,
                 wls_lambda: float = None,
                 wls_sigma: float = None):
        self.mono_frames = mono_frames

        self.multiplier = 255.0 / max_disp
        self.fps = fps

        self.colorize = colorize
        self.colormap = colormap

        self.use_wls_filter = use_wls_filter

        # Prefer to use WLS level if set, otherwise use lambda and sigma
        if wls_level and use_wls_filter:
            print(f'Using WLS level: {wls_level.name} (lambda: {wls_level.value[0]}, sigma: {wls_level.value[1]})')
            self.wls_lambda = wls_level.value[0]
            self.wls_sigma = wls_level.value[1]
        else:
            self.wls_lambda = wls_lambda
            self.wls_sigma = wls_sigma

        if self.use_wls_filter:
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

        self.msgs = dict()

        XoutFrames.__init__(self, frames=disparity_frames, fps=fps)
        XoutClickable.__init__(self, decay_step=int(self.fps))

    def visualize(self, packet: DepthPacket):
        frame = packet.frame
        disparity_frame = (frame * self.multiplier).astype(np.uint8)

        stereo_config = self._visualizer.config.stereo

        if self.use_wls_filter or stereo_config.wls_filter:
            self.wls_filter.setLambda(self.wls_lambda or stereo_config.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma or stereo_config.wls_sigma)
            disparity_frame = self.wls_filter.filter(disparity_frame, packet.mono_frame.getCvFrame())

        colorize = self.colorize if self.colorize is not None else stereo_config.colorize
        colormap = self.colormap or stereo_config.colormap
        if colorize == StereoColor.GRAY:
            packet.frame = disparity_frame
        elif colorize == StereoColor.RGB:
            packet.frame = cv2.applyColorMap(disparity_frame, colormap or cv2.COLORMAP_JET)
        elif colorize == StereoColor.RGBD:
            packet.frame = cv2.applyColorMap(
                (disparity_frame * 0.5 + packet.mono_frame.getCvFrame() * 0.5).astype(np.uint8),
                colormap or cv2.COLORMAP_JET
            )

        if self._visualizer.config.output.clickable:
            cv2.namedWindow(self.name)
            cv2.setMouseCallback(self.name, self.on_click_callback, param=[disparity_frame])

            if self.buffer:
                x, y = self.buffer[2]
                self._visualizer.add_circle(coords=(x, y), radius=3, color=(255, 255, 255), thickness=-1)
                self._visualizer.add_text(
                    text=f'{self.buffer[1]}',
                    coords=(x, y - 10)
                )

        super().visualize(packet)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.mono_frames]

    def new_msg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams:
            return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()

        if name == self.frames.name:
            self.msgs[seq][name] = msg
        elif name == self.mono_frames.name:
            self.msgs[seq][name] = msg
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if len(self.msgs[seq]) == len(self.xstreams()):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            packet = DepthPacket(
                self.get_packet_name(),
                self.msgs[seq][self.frames.name],
                self.msgs[seq][self.mono_frames.name],
                self._visualizer
            )
            self.queue.put(packet, block=False)

            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs


# TODO can we merge XoutDispariry and XoutDepth?
class XoutDepth(XoutFrames, XoutClickable):
    def __init__(self,
                 device: dai.Device,
                 frames: StreamXout,
                 fps: float,
                 mono_frames: StreamXout,
                 colorize: StereoColor = None,
                 colormap: int = None,
                 use_wls_filter: bool = None,
                 wls_level: 'WLSLevel' = None,
                 wls_lambda: float = None,
                 wls_sigma: float = None):
        self.mono_frames = mono_frames
        XoutFrames.__init__(self, frames=frames, fps=fps)
        XoutClickable.__init__(self, decay_step=int(self.fps))

        self.name = 'Depth'
        self.fps = fps
        self.device = device

        self.colorize = colorize
        self.colormap = colormap

        self.use_wls_filter = use_wls_filter

        # Prefer to use WLS level if set, otherwise use lambda and sigma
        if wls_level and use_wls_filter:
            print(f'Using WLS level: {wls_level.name} (lambda: {wls_level.value[0]}, sigma: {wls_level.value[1]})')
            self.wls_lambda = wls_level.value[0]
            self.wls_sigma = wls_level.value[1]
        else:
            self.wls_lambda = wls_lambda
            self.wls_sigma = wls_sigma

        self.wls_filter = None
        self.msgs = dict()

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        super().setup_visualize(visualizer, visualizer_enabled, name)

        if self._visualizer.config.stereo.wls_filter or self.use_wls_filter:
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

    def visualize(self, packet: DepthPacket):
        depth_frame = packet.imgFrame.getFrame()

        stereo_config = self._visualizer.config.stereo

        if self.use_wls_filter or stereo_config.wls_filter:
            self.wls_filter.setLambda(stereo_config.wls_lambda)
            self.wls_filter.setSigmaColor(stereo_config.wls_sigma)
            depth_frame = self.wls_filter.filter(depth_frame, packet.mono_frame.getCvFrame())

        depth_frame_color = cv2.normalize(depth_frame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
        depth_frame_color = cv2.equalizeHist(depth_frame_color)

        colorize = self.colorize or stereo_config.colorize
        colormap = self.colormap or stereo_config.colormap
        if colorize == StereoColor.GRAY:
            packet.frame = depth_frame_color
        elif colorize == StereoColor.RGB:
            packet.frame = cv2.applyColorMap(depth_frame_color, colormap or cv2.COLORMAP_JET)
        elif colorize == StereoColor.RGBD:
            packet.frame = cv2.applyColorMap(
                (depth_frame_color * 0.5 + packet.mono_frame.getCvFrame() * 0.5).astype(np.uint8),
                stereo_config.colormap or cv2.COLORMAP_JET
            )

        if self._visualizer.config.output.clickable:
            cv2.namedWindow(self.name)
            cv2.setMouseCallback(self.name, self.on_click_callback, param=[depth_frame])

            if self.buffer:
                x, y = self.buffer[2]
                self._visualizer.add_circle(coords=(x, y), radius=3, color=(255, 255, 255), thickness=-1)
                self._visualizer.add_text(
                    text=f'{self.buffer[1] / 10} cm',
                    coords=(x, y - 10)
                )

        super().visualize(packet)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.mono_frames]

    def new_msg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams:
            return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()

        if name == self.frames.name:
            self.msgs[seq][name] = msg
        elif name == self.mono_frames.name:
            self.msgs[seq][name] = msg
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if len(self.msgs[seq]) == len(self.xstreams()):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            packet = DepthPacket(
                self.get_packet_name(),
                self.msgs[seq][self.frames.name],
                self.msgs[seq][self.mono_frames.name],
                self._visualizer
            )
            self.queue.put(packet, block=False)

            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs


class XoutSeqSync(XoutBase, SequenceNumSync):
    def xstreams(self) -> List[StreamXout]:
        return self.streams

    def __init__(self, streams: List[StreamXout]):
        self.streams = streams
        # Save StreamXout before initializing super()!
        XoutBase.__init__(self)
        SequenceNumSync.__init__(self, len(streams))
        self.msgs = dict()

    @abstractmethod
    def package(self, msgs: List):
        raise NotImplementedError('XoutSeqSync is an abstract class, you need to override package() method!')

    def new_msg(self, name: str, msg) -> None:
        # Ignore frames that we aren't listening for
        if name not in self._streams: return

        synced = self.sync(msg.getSequenceNum(), name, msg)
        if synced:
            self.package(synced)


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
            # np.array of (b,g,r), 0..1
            colors = np.array(distinctipy.get_colors(n_colors=n_colors, rng=123123, pastel_factor=0.5))[..., ::-1]
            colors = [distinctipy.get_rgb256(clr) for clr in colors]  # List of (b,g,r), 0..255
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

        self.normalizer = NormalizeBoundingBox(det_nn._size, det_nn._ar_resize_mode)
        try:
            self._frame_shape = self.det_nn._input.node.getPreviewSize()
        except AttributeError:
            self._frame_shape = self.det_nn._input.stream_size  # Replay

        self._frame_shape = np.array(self._frame_shape)[::-1]

        self.segmentation_colormap = None

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        super().setup_visualize(visualizer, visualizer_enabled, name)

    def on_callback(self, packet: Union[DetectionPacket, TrackerPacket]):
        if self._visualizer.frame_shape is None:
            if packet.frame.ndim == 1:
                self._visualizer.frame_shape = self._frame_shape
            else:
                self._visualizer.frame_shape = packet.frame.shape

        # Add detections to packet
        if isinstance(packet.img_detections, dai.ImgDetections) \
                or isinstance(packet.img_detections, dai.SpatialImgDetections) \
                or isinstance(packet.img_detections, Detections):
            for detection in packet.img_detections.detections:
                d = _Detection()
                d.img_detection = detection
                d.label = self.labels[detection.label][0] if self.labels else str(detection.label)
                d.color = self.labels[detection.label][1] if self.labels else (255, 255, 255)
                bbox = self.normalizer.normalize(
                    frame=np.zeros(self._visualizer.frame_shape, dtype=bool),
                    bbox=(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                )
                d.top_left = (int(bbox[0]), int(bbox[1]))
                d.bottom_right = (int(bbox[2]), int(bbox[3]))
                packet.detections.append(d)

            # Add detections to visualizer
            self._visualizer.add_detections(
                packet.img_detections.detections,
                self.normalizer,
                self.labels,
                is_spatial=packet._is_spatial_detection()
            )
        elif isinstance(packet.img_detections, ImgLandmarks):
            all_landmarks = packet.img_detections.landmarks
            all_landmarks_indices = packet.img_detections.landmarks_indices
            colors = packet.img_detections.colors
            for landmarks, indices in zip(all_landmarks, all_landmarks_indices):
                for i, landmark in enumerate(landmarks):
                    l = self.normalizer.normalize(frame=np.zeros(self._visualizer.frame_shape, dtype=bool),
                                                  bbox=landmark)
                    idx = indices[i]

                    self._visualizer.add_line(pt1=tuple(l[0]), pt2=tuple(l[1]), color=colors[idx], thickness=4)
                    self._visualizer.add_circle(coords=tuple(l[0]), radius=8, color=colors[idx], thickness=-1)
                    self._visualizer.add_circle(coords=tuple(l[1]), radius=8, color=colors[idx], thickness=-1)
        elif isinstance(packet.img_detections, SemanticSegmentation):
            # Generate colormap if not already generated
            if self.segmentation_colormap is None:
                n_classes = len(self.labels) if self.labels else 8
                self.segmentation_colormap = self._generate_colors(n_classes)

            mask = np.array(packet.img_detections.mask).astype(np.uint8)
            try:
                colorized_mask = np.array(self.segmentation_colormap)[mask]
            except IndexError:
                unique_classes = np.unique(mask)
                max_class = np.max(unique_classes)
                new_colors = self._generate_colors(max_class - len(self.segmentation_colormap) + 1)
                self.segmentation_colormap.extend(new_colors)

            bbox = None
            if self.normalizer.resize_mode == ResizeMode.LETTERBOX:
                bbox = self.normalizer.get_letterbox_bbox(packet.frame, normalize=True)
                input_h, input_w = self.normalizer.aspect_ratio
                resize_bbox = bbox[0] * input_w, bbox[1] * input_h, bbox[2] * input_w, bbox[3] * input_h
                resize_bbox = np.int0(resize_bbox)
            else:
                resize_bbox = self.normalizer.normalize(frame=np.zeros(self._visualizer.frame_shape, dtype=bool),
                                                        bbox=bbox or (0., 0., 1., 1.))

            x1, y1, x2, y2 = resize_bbox
            h, w = packet.frame.shape[:2]
            # Stretch mode
            if self.normalizer.resize_mode == ResizeMode.STRETCH:
                colorized_mask = cv2.resize(colorized_mask, (w, h))
            elif self.normalizer.resize_mode == ResizeMode.LETTERBOX:
                colorized_mask = cv2.resize(colorized_mask[y1:y2, x1:x2], (w, h))
            else:
                padded_mask = np.zeros((h, w, 3), dtype=np.uint8)
                resized_mask = cv2.resize(colorized_mask, (x2 - x1, y2 - y1))
                padded_mask[y1:y2, x1:x2] = resized_mask
                colorized_mask = padded_mask

            self._visualizer.add_mask(colorized_mask, alpha=0.5)

    def package(self, msgs: Dict):
        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        decode_fn = self.det_nn._decode_fn or (self.det_nn._handler.decode if self.det_nn._handler else None)
        packet = DetectionPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            msgs[self.nn_results.name] if decode_fn is None else decode_fn(msgs[self.nn_results.name]),
            self._visualizer
        )

        self.queue.put(packet, block=False)

    def _generate_colors(self, n_colors, exclude=None):
        colors = distinctipy.get_colors(n_colors, exclude / 255 if exclude else None,
                                        rng=11, pastel_factor=0.3, n_attempts=100)
        rgb_colors = np.array(colors) * 255
        return rgb_colors.astype(np.uint8)


class XoutSpatialBbMappings(XoutSeqSync, XoutFrames):
    def __init__(self, device: dai.Device, frames: StreamXout, configs: StreamXout):
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
            size = (packet.imgFrame.getWidth(), packet.imgFrame.getHeight())
            self.factor = calc_disp_multiplier(self.device, size)

        depth = np.array(packet.imgFrame.getFrame())
        with np.errstate(divide='ignore'):
            disp = (self.factor / depth).astype(np.uint8)

        packet.frame = colorize_disparity(disp, multiplier=self.multiplier)
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


class XoutTracker(XoutNnResults):
    buffer_size: int = 10

    def __init__(self, det_nn, frames: StreamXout, tracklets: StreamXout):
        super().__init__(det_nn, frames, tracklets)
        self.buffer = []
        self.name = 'Object Tracker'
        self.lost_counter = {}

    def on_callback(self, packet: Union[DetectionPacket, TrackerPacket]):
        try:
            if packet._is_spatial_detection():
                spatial_points = [packet._get_spatials(det.srcImgDetection)
                                  for det in
                                  packet.daiTracklets.tracklets]
            else:
                spatial_points = None
        except IndexError:
            spatial_points = None

        if self._visualizer.frame_shape is None:
            if packet.frame.ndim == 1:
                self._visualizer.frame_shape = self._frame_shape
            else:
                self._visualizer.frame_shape = packet.frame.shape

        blacklist = set()
        threshold = self._visualizer.config.tracking.deletion_lost_threshold
        for i, tracklet in enumerate(packet.daiTracklets.tracklets):
            if tracklet.status == dai.Tracklet.TrackingStatus.LOST:
                self.lost_counter[tracklet.id] += 1
            elif tracklet.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.lost_counter[tracklet.id] = 0

            if tracklet.id in self.lost_counter and self.lost_counter[tracklet.id] >= threshold:
                blacklist.add(tracklet.id)

        filtered_tracklets = [tracklet for tracklet in packet.daiTracklets.tracklets if tracklet.id not in blacklist]
        self._visualizer.add_detections(filtered_tracklets,
                                        self.normalizer,
                                        self.labels,
                                        spatial_points=spatial_points)

        # Add to local storage
        self.buffer.append(packet)
        if self.buffer_size < len(self.buffer):
            self.buffer.pop(0)

        self._visualizer.add_trail(
            tracklets=[t for p in self.buffer for t in p.daiTracklets.tracklets if t.id not in blacklist],
            label_map=self.labels
        )

        # Add trail id
        h, w = packet.frame.shape[:2]
        for tracklet in filtered_tracklets:
            det = tracklet.srcImgDetection
            bbox = (w * det.xmin, h * det.ymin, w * det.xmax, h * det.ymax)
            bbox = tuple(map(int, bbox))
            self._visualizer.add_text(
                f'ID: {tracklet.id}',
                bbox=bbox,
                position=TextPosition.MID
            )

    def package(self, msgs: Dict):
        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full
        packet = TrackerPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            msgs[self.nn_results.name],
            self._visualizer
        )
        self.queue.put(packet, block=False)


class XoutNnH26x(XoutNnResults, XoutH26x):
    name: str = "H26x NN Results"
    # Streams
    frames: StreamXout
    nn_results: StreamXout

    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout,
                 color: bool,
                 profile: dai.VideoEncoderProperties.Profile,
                 fps: float,
                 frame_shape: Tuple[int, ...]):
        self.nn_results = nn_results

        XoutH26x.__init__(self, frames, color, profile, fps, frame_shape)
        XoutNnResults.__init__(self, det_nn, frames, nn_results)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results]

    def visualize(self, packet: FramePacket):
        decoded_frame = XoutH26x.decode_frame(self, packet)
        if decoded_frame is None:
            return

        packet.frame = decoded_frame
        XoutNnResults.visualize(self, packet)


class XoutNnMjpeg(XoutNnResults, XoutMjpeg):
    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 nn_results: StreamXout,
                 color: bool,
                 lossless: bool,
                 fps: float,
                 frame_shape: Tuple[int, ...]):
        self.nn_results = nn_results
        XoutMjpeg.__init__(self, frames, color, lossless, fps, frame_shape)
        XoutNnResults.__init__(self, det_nn, frames, nn_results)

    def visualize(self, packet: FramePacket):
        packet.frame = XoutMjpeg.decode_frame(self, packet)
        XoutNnResults.visualize(self, packet)


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

        self.whitelist_labels: Optional[List[int]] = None
        self.scale_bb: Optional[Tuple[int, int]] = None

        conf = det_nn._multi_stage_config  # No types due to circular import...
        if conf is not None:
            self.labels = conf._labels
            self.scale_bb = conf.scale_bb

        self.device = device
        self.input_queue_name = input_queue_name
        self.input_queue = None
        self.input_cfg_queue = None

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


class XoutIMU(XoutBase):
    def __init__(self, imu_xout: StreamXout):
        self.imu_out = imu_xout
        self.packets = []
        self.start_time = 0.0

        self.fig = None
        self.axes = None
        self.acceleration_lines = []
        self.gyroscope_lines = []

        self.acceleration_buffer = []
        self.gyroscope_buffer = []

        super().__init__()
        self.name = 'IMU'

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None, _=None):
        from matplotlib import pyplot as plt

        self._visualizer = visualizer
        self._visualizer_enabled = visualizer_enabled
        self.name = name or self.name

        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
        labels = ['x', 'y', 'z']

        for i in range(3):
            self.acceleration_lines.append(self.axes[0].plot([], [], label=f'Acceleration {labels[i]}')[0])
            self.axes[0].set_ylabel('Acceleration (m/s^2)')
            self.axes[0].set_xlabel('Time (s)')
            self.axes[0].legend()

        for i in range(3):
            self.gyroscope_lines.append(self.axes[1].plot([], [], label=f'Gyroscope {labels[i]}')[0])
            self.axes[1].set_ylabel('Gyroscope (rad/s)')
            self.axes[1].set_xlabel('Time (s)')
            self.axes[1].legend()

    def visualize(self, packet: IMUPacket):
        if self.start_time == 0.0:
            self.start_time = packet.data[0].acceleroMeter.timestamp.get()

        acceleration_x = [el.acceleroMeter.x for el in packet.data]
        acceleration_z = [el.acceleroMeter.y for el in packet.data]
        acceleration_y = [el.acceleroMeter.z for el in packet.data]

        t_acceleration = [(el.acceleroMeter.timestamp.get() - self.start_time).total_seconds() for el in packet.data]

        # Keep only last 100 values
        if len(self.acceleration_buffer) > 100:
            self.acceleration_buffer.pop(0)

        self.acceleration_buffer.append([t_acceleration, acceleration_x, acceleration_y, acceleration_z])

        gyroscope_x = [el.gyroscope.x for el in packet.data]
        gyroscope_y = [el.gyroscope.y for el in packet.data]
        gyroscope_z = [el.gyroscope.z for el in packet.data]

        t_gyroscope = [(el.gyroscope.timestamp.get() - self.start_time).total_seconds() for el in packet.data]

        # Keep only last 100 values
        if len(self.gyroscope_buffer) > 100:
            self.gyroscope_buffer.pop(0)

        self.gyroscope_buffer.append([t_gyroscope, gyroscope_x, gyroscope_y, gyroscope_z])

        # Plot acceleration
        for i in range(3):
            self.acceleration_lines[i].set_xdata([el[0] for el in self.acceleration_buffer])
            self.acceleration_lines[i].set_ydata([el[i + 1] for el in self.acceleration_buffer])

        self.axes[0].set_xlim(self.acceleration_buffer[0][0][0], t_acceleration[-1])
        self.axes[0].set_ylim(-20, 20)

        # Plot gyroscope
        for i in range(3):
            self.gyroscope_lines[i].set_xdata([el[0] for el in self.gyroscope_buffer])
            self.gyroscope_lines[i].set_ydata([el[i + 1] for el in self.gyroscope_buffer])

        self.axes[1].set_xlim(self.gyroscope_buffer[0][0][0], t_acceleration[-1])
        self.axes[1].set_ylim(-20, 20)

        self.fig.canvas.draw()

        # Convert plot to numpy array
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        packet.frame = img

        if self.callback:  # Don't display frame, call the callback
            self.callback(packet)
        else:
            packet.frame = self._visualizer.draw(packet.frame)
            cv2.imshow(self.name, packet.frame)

    def xstreams(self) -> List[StreamXout]:
        return [self.imu_out]

    def new_msg(self, name: str, msg: dai.IMUData) -> None:
        if name not in self._streams:
            return

        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        packet = IMUPacket(msg.packets)

        self.queue.put(packet, block=False)

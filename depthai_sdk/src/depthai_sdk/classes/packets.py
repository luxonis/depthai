from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Tuple, List, Union, Optional

import depthai as dai
import numpy as np
from depthai_sdk.visualize.bbox import BoundingBox
try:
    import cv2
except ImportError:
    cv2 = None


class _Detection:
    # Original ImgDetection
    img_detection: dai.ImgDetection
    label_str: str
    color: Tuple[int, int, int]

    # Normalized bounding box
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

    def centroid(self) -> Tuple[int, int]:
        return (
            int((self.bottom_right[0] + self.top_left[0]) / 2),
            int((self.bottom_right[1] + self.top_left[1]) / 2),
        )

    def get_bbox(self) -> Tuple[float, float, float, float]:
        return self.img_detection.xmin, self.img_detection.ymin, self.img_detection.xmax, self.img_detection.ymax


class _TrackingDetection(_Detection):
    tracklet: dai.Tracklet
    speed: float = 0.0  # m/s
    speed_kmph: float = 0.0  # km/h
    speed_mph: float = 0.0  # mph


class _TwoStageDetection(_Detection):
    nn_data: dai.NNData


class BasePacket(ABC):
    """
    Base class for all packets.
    """

    def __init__(self, name: str):
        self.name = name

    def prepare_visualizer_objects(self, visualizer: 'Visualizer') -> None:
        """
        Prepare visualizer objects (boxes, lines, text, etc.), so visualizer can draw them on the frame.

        Args:
            visualizer: Visualizer object.
        """
        pass

    @abstractmethod
    def get_timestamp(self) -> timedelta:
        raise NotImplementedError()

    @abstractmethod
    def get_sequence_num(self) -> int:
        raise NotImplementedError()

class NNDataPacket(BasePacket):
    """
    Contains only dai.NNData message
    """
    def __init__(self, name: str, nn_data: dai.NNData):
        self.msg = nn_data
        super().__init__(name)

class FramePacket(BasePacket):
    """
    Contains only dai.ImgFrame message and cv2 frame, which is used by visualization logic.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 frame: Optional[np.ndarray]):
        self.msg = msg
        self.frame = frame
        super().__init__(name)

    def get_timestamp(self) -> timedelta:
        return self.msg.getTimestampDevice(dai.CameraExposureOffset.MIDDLE)

    def get_sequence_num(self) -> int:
        return self.msg.getSequenceNum()


class DepthPacket(FramePacket):
    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 mono_frame: Optional[dai.ImgFrame],
                 depth_map: Optional[np.ndarray] = None):
        super().__init__(name=name,
                         msg=img_frame,
                         frame=img_frame.getCvFrame() if cv2 else None)
        self.mono_frame = mono_frame
        self.depth_map = depth_map

class PointcloudPacket(DepthPacket):
    def __init__(self,
                 name: str,
                 points: np.ndarray,
                 depth_map: dai.ImgFrame,
                 colorize_frame: Optional[np.ndarray]):
        self.points = points
        super().__init__(name=name,
                        img_frame=depth_map,
                        mono_frame=colorize_frame,
                        depth_map=depth_map.getFrame())


class SpatialBbMappingPacket(FramePacket):
    """
    Output from Spatial Detection nodes - depth frame + bounding box mappings. Inherits FramePacket.
    """
    spatials: dai.SpatialImgDetections

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 spatials: dai.SpatialImgDetections):
        super().__init__(name=name,
                         msg=msg,
                         frame=msg.getFrame() if cv2 else None)
        self.spatials = spatials


class DetectionPacket(FramePacket):
    """
    Output from Detection Network nodes - image frame + image detections. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 img_detections: Union[dai.ImgDetections, dai.SpatialImgDetections]):
        super().__init__(name=name,
                         msg=msg,
                         frame=msg.getCvFrame() if cv2 else None)
        self.img_detections = img_detections
        self.detections: List[_Detection] = []

    def _is_spatial_detection(self) -> bool:
        return isinstance(self.img_detections, dai.SpatialImgDetections)

    def _add_detection(self, img_det: dai.ImgDetection, bbox: np.ndarray, txt: str, color) -> None:
        det = _Detection()
        det.img_detection = img_det
        det.label_str = txt
        det.color = color
        det.top_left = (bbox[0], bbox[1])
        det.bottom_right = (bbox[2], bbox[3])
        self.detections.append(det)

    def prepare_visualizer_objects(self, visualizer: 'Visualizer') -> None:
        # Convert Grayscale to BGR
        if len(self.frame.shape) == 2:
            self.frame = np.dstack((self.frame, self.frame, self.frame))

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


class TrackerPacket(FramePacket):
    """
    Output of Object Tracker node. Tracklets + Image frame. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 tracklets: dai.Tracklets):
        super().__init__(name=name,
                         msg=msg,
                         frame=msg.getCvFrame() if cv2 else None)
        self.detections: List[_TrackingDetection] = []
        self.daiTracklets = tracklets

    def _add_detection(self, img_det: dai.ImgDetection, bbox: np.ndarray, txt: str, color):
        det = _TrackingDetection()
        det.img_detection = img_det
        det.label_str = txt
        det.color = color
        det.top_left = (bbox[0], bbox[1])
        det.bottom_right = (bbox[2], bbox[3])
        self.detections.append(det)

    def _is_spatial_detection(self) -> bool:
        coords = self.daiTracklets.tracklets[0].spatialCoordinates
        return coords.x != 0.0 or coords.y != 0.0 or coords.z != 0.0

    def _get_spatials(self, det: dai.ImgDetection) -> dai.Point3f:
        # Not the cleanest solution, but oh well
        for t in self.daiTracklets.tracklets:
            if t.srcImgDetection == det:
                return t.spatialCoordinates


class TwoStagePacket(DetectionPacket):
    """
    Output of 2-stage NN pipeline; Image frame, Image detections and multiple NNData results. Inherits DetectionPacket.
    """

    def __init__(self, name: str,
                 msg: dai.ImgFrame,
                 img_detections: dai.ImgDetections,
                 nn_data: List[dai.NNData],
                 labels: List[int]):
        super().__init__(name=name,
                         msg=msg,
                         img_detections=img_detections)
        self.frame = self.msg.getCvFrame() if cv2 else None
        self.nnData = nn_data
        self.labels = labels
        self._cntr = 0

    def _add_detection(self, img_det: dai.ImgDetection, bbox: np.ndarray, txt: str, color):
        det = _TwoStageDetection()
        det.img_detection = img_det
        det.color = color
        det.top_left = (bbox[0], bbox[1])
        det.bottom_right = (bbox[2], bbox[3])

        # Append the second stage NN result to the detection
        if self.labels is None or img_det.label in self.labels:
            det.nn_data = self.nnData[self._cntr]
            self._cntr += 1

        self.detections.append(det)


class IMUPacket(BasePacket):
    def __init__(self, name, packet: dai.IMUPacket):
        self.packet = packet
        super().__init__(name)

    def __str__(self):
        accelerometer_str = 'Accelerometer [m/s^2]: (x: %.2f, y: %.2f, z: %.2f)' % (
            self.packet.acceleroMeter.x,
            self.packet.acceleroMeter.y,
            self.packet.acceleroMeter.z
        )

        gyroscope_str = 'Gyroscope [rad/s]: (x: %.2f, y: %.2f, z: %.2f)' % (
            self.packet.gyroscope.x,
            self.packet.gyroscope.y,
            self.packet.gyroscope.z
        )

        return f'IMU Packet: {accelerometer_str} {gyroscope_str}'

    def _get_imu_report(self) -> dai.IMUReport:
        if self.packet.acceleroMeter is not None:
            return self.packet.acceleroMeter
        elif self.packet.gyroscope is not None:
            return self.packet.gyroscope
        elif self.packet.magneticField is not None:
            return self.packet.magneticField
        elif self.packet.rotationVector is not None:
            return self.packet.rotationVector
        raise RuntimeError('Unknown IMU packet type')

    def get_timestamp(self) -> timedelta:
        return self._get_imu_report().getTimestampDevice()

    def get_sequence_num(self) -> int:
        return self._get_imu_report().getSequenceNum()

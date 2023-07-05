from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Sequence, Tuple, List, Union, Optional, Dict
from depthai_sdk.classes import Detections, ImgLandmarks, SemanticSegmentation
import depthai as dai
import numpy as np
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.configs import TextPosition
from depthai_sdk.visualize.visualizer import Visualizer
from depthai_sdk.classes.nn_results import Detection, TrackingDetection, TwoStageDetection

try:
    import cv2
except ImportError:
    cv2 = None


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
    def get_timestamp(self) -> timedelta:
        return self.msg.getTimestamp()
    def get_sequence_num(self) -> int:
        return self.msg.getTimestampDevice()

class FramePacket(BasePacket):
    """
    Contains only dai.ImgFrame message and cv2 frame, which is used by visualization logic.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame):
        self.msg = msg
        super().__init__(name)

    def get_timestamp(self) -> timedelta:
        return self.msg.getTimestampDevice(dai.CameraExposureOffset.MIDDLE)

    def get_sequence_num(self) -> int:
        return self.msg.getSequenceNum()

    def decode(self) -> Optional[np.ndarray]:
        return self.msg.getCvFrame() if cv2 else None

    def get_size(self) -> Tuple[int, int]:
        return self.msg.getWidth(), self.msg.getHeight()

class H26xPacket(FramePacket):
    def __init__(self,
                name: str,
                msg: dai.ImgFrame,
                codec,
                is_color: bool
                ):
        super().__init__(name=name, msg=msg)
        self.codec = codec
        self.is_color = is_color

    def decode(self) -> Optional[np.ndarray]:
        if self.codec is None:
            raise ImportError('av is not installed. Please install it with `pip install av`')

        enc_packets = self.codec.parse(self.msg.getData())
        if len(enc_packets) == 0:
            return None

        frames = self.codec.decode(enc_packets[-1])
        if not frames:
            return None

        frame = frames[0].to_ndarray(format='bgr24')

        if not self.is_color:
            # Convert to grayscale
            frame = frame[:, :, 0]
        return frame

class MjpegPacket(FramePacket):
    def __init__(self,
                name: str,
                msg: dai.ImgFrame,
                is_color: bool,
                is_lossless: bool,
                ):
        self.is_lossless = is_lossless
        self.is_color = is_color
        super().__init__(name=name, msg=msg)

    def decode(self) -> np.ndarray:
        if self.is_lossless:
            raise NotImplementedError('Lossless MJPEG decoding is not supported!')
        if cv2 is None:
            raise ImportError('cv2 is not installed. Please install it with `pip install opencv-python`')
        flag = cv2.IMREAD_COLOR if self.is_color else cv2.IMREAD_GRAYSCALE
        return cv2.imdecode(self.msg.getData(), flag)


class DepthPacket(FramePacket):
    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 mono_frame: Optional[dai.ImgFrame],
                 depth_map: Optional[np.ndarray] = None):
        super().__init__(name=name, msg=img_frame)
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
    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 spatials: dai.SpatialImgDetections):
        super().__init__(name=name,
                         msg=msg)
        self.spatials = spatials


class NnOutputPacket(FramePacket):
    """
    NN result + image frame. Inherits FramePacket.
    """
    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 nn_data: dai.NNData,
                 bbox: BoundingBox
                 ):
        super().__init__(name=name,
                         msg=msg)
        self.nn_data = nn_data
        self.bbox = bbox

class ImgLandmarksPacket(NnOutputPacket):
    """
    Output from Landmarks Estimation nodes - image frame + landmarks. Inherits NnOutputPacket.
    """
    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 nn_data: dai.NNData,
                 landmarks: ImgLandmarks,
                 bbox: BoundingBox):
        super().__init__(name=name,
                         msg=msg,
                         nn_data=nn_data,
                         bbox=bbox)
        self.landmarks = landmarks

    def prepare_visualizer_objects(self, vis: Visualizer) -> None:
        all_landmarks = self.landmarks.landmarks
        all_landmarks_indices = self.landmarks.landmarks_indices
        colors = self.landmarks.colors
        w, h = self.get_size()
        for landmarks, indices in zip(all_landmarks, all_landmarks_indices):
            for i, landmark in enumerate(landmarks):
                # Map normalized coordinates to frame coordinates
                l = [(int(point[0] * w), int(point[1] * h)) for point in landmark]
                idx = indices[i]

                vis.add_line(pt1=tuple(l[0]), pt2=tuple(l[1]), color=colors[idx], thickness=4)
                vis.add_circle(coords=tuple(l[0]), radius=8, color=colors[idx], thickness=-1)
                vis.add_circle(coords=tuple(l[1]), radius=8, color=colors[idx], thickness=-1)

class SemanticSegmentationPacket(NnOutputPacket):
    """
    Output from Semantic Segmentation nodes - image frame + segmentation mask. Inherits NnOutputPacket.
    """
    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 nn_data: dai.NNData,
                 segmentation: SemanticSegmentation,
                 bbox: BoundingBox):
        super().__init__(name=name,
                         msg=msg,
                         nn_data=nn_data,
                         bbox=bbox)
        self.segmentation = segmentation

    def prepare_visualizer_objects(self, vis: Visualizer) -> None:
        raise NotImplementedError('Semantic segmentation visualization is not implemented yet!')

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

class DetectionPacket(FramePacket):
    """
    Output from Detection Network nodes - image frame + image detections. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 dai_msg: Union[dai.ImgDetections, dai.SpatialImgDetections, dai.NNData],
                 bbox: BoundingBox,
                ):

        super().__init__(name=name,
                         msg=msg)

        self.img_detections = dai_msg
        self.bbox = bbox
        self.detections: List[Detection] = []

    def _is_spatial_detection(self) -> bool:
        return isinstance(self.img_detections, dai.SpatialImgDetections)

    def prepare_visualizer_objects(self, vis: Visualizer) -> None:
        # Add detections to packet
        if isinstance(self.img_detections, dai.ImgDetections) \
                or isinstance(self.img_detections, dai.SpatialImgDetections) \
                or isinstance(self.img_detections, Detections):

            for detection in self.detections:
                # Add detections to visualizer
                vis.add_bbox(
                    bbox=self.bbox.get_relative_bbox(detection.bbox),
                    label=detection.label_str,
                    color=detection.color,
                )


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

            # vis.add_mask(colorized_mask, alpha=0.5)


class TrackerPacket(FramePacket):
    """
    Output of Object Tracker node. Tracklets + Image frame. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 tracklets: dai.Tracklets):
        super().__init__(name=name,
                         msg=msg)
        self.detections: List[TrackingDetection] = []
        self.daiTracklets = tracklets

    def _add_detection(self, img_det: dai.ImgDetection, bbox: np.ndarray, txt: str, color):
        det = TrackingDetection()
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

    def prepare_visualizer_objects(self, visualizer: Visualizer) -> None:
        # self._add_tracklet_visualization()

    # def _add_tracklet_visualization(self, packet, spatial_points, tracklet2speed):
        h, w = self.msg.getHeight(), self.msg.getWidth()
        filtered_tracklets = [tracklet for tracklet in self.daiTracklets.tracklets if
                              tracklet.id not in self.blacklist]
        self.frame

        norm_bbox = BoundingBox().resize_to_aspect_ratio(self.frame.shape, self._nn_size, self._resize_mode)

        visualizer.add_detections(detections=filtered_tracklets,
                                    normalizer=norm_bbox,
                                    label_map=self.labels,
                                    spatial_points=spatial_points)

        # Add tracking ids
        for tracklet in filtered_tracklets:
            det = tracklet.srcImgDetection
            bbox = (w * det.xmin, h * det.ymin, w * det.xmax, h * det.ymax)
            bbox = tuple(map(int, bbox))
            visualizer.add_text(
                f'ID: {tracklet.id}',
                bbox=bbox,
                position=TextPosition.MID
            )

            if visualizer.config.tracking.show_speed and tracklet.id in tracklet2speed:
                speed = tracklet2speed[tracklet.id]
                speed = f'{speed:.1f} m/s\n{speed * 3.6:.1f} km/h'
                bbox = tracklet.srcImgDetection
                bbox = (int(w * bbox.xmin), int(h * bbox.ymin), int(w * bbox.xmax), int(h * bbox.ymax))

                visualizer.add_text(
                    speed,
                    bbox=bbox,
                    position=TextPosition.TOP_RIGHT,
                    outline=True
                )

        # Add tracking lines
        visualizer.add_trail(
            tracklets=[t for p in self.buffer for t in p.daiTracklets.tracklets if t.id not in self.blacklist],
            label_map=self.labels,
            bbox=norm_bbox,
        )


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
        det = TwoStageDetection()
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
    def __init__(self, name, packet: dai.IMUPacket, rotation = None):
        self.packet = packet
        super().__init__(name)

        self.acceleroMeter = packet.acceleroMeter
        self.gyroscope = packet.gyroscope
        self.magneticField = packet.magneticField
        self.rotationVector = rotation if rotation is not None else packet.rotationVector

        # Check which reports are available
        self.available_reports: Dict[str, dai.IMUReport] = {}
        for i, val in enumerate([self.acceleroMeter, self.gyroscope, self.magneticField, self.rotationVector]):
            if (i==3 and rotation) or val.getTimestampDevice() != timedelta(0):
                self.available_reports[val.__class__.__name__] = val

    def get_imu_vals(self) -> Tuple[Sequence, Sequence, Sequence, Sequence]:
        """
        Returns imu values in a tuple. Returns in format (accelerometer_values, gyroscope_values, quaternion, magnetometer_values)
        """
        return (
            [self.acceleroMeter.x, self.acceleroMeter.y, self.acceleroMeter.z],
            [self.gyroscope.x, self.gyroscope.y, self.gyroscope.z],
            [self.rotationVector.i, self.rotationVector.j, self.rotationVector.k, self.rotationVector.real],
            [self.magneticField.x, self.magneticField.y, self.magneticField.z]
        )

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
        """
        Get the first available IMU report
        """
        for name, val in self.available_reports.items():
            return val

    def get_timestamp(self) -> timedelta:
        return self._get_imu_report().getTimestampDevice()

    def get_sequence_num(self) -> int:
        return self._get_imu_report().getSequenceNum()

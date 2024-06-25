from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Sequence, Tuple, List, Union, Optional, Dict, Callable

import depthai as dai
import numpy as np

from depthai_sdk.classes import ImgLandmarks, SemanticSegmentation
from depthai_sdk.classes.nn_results import Detection, TrackingDetection, TwoStageDetection
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.configs import StereoColor, TextPosition
from depthai_sdk.visualize.visualizer import Visualizer

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
                 msg: dai.ImgFrame,
                 ):
        self.msg = msg
        self._get_codec = None
        self.__frame = None
        super().__init__(name)

    @property
    def frame(self):
        if self.__frame is None:
            self.__frame = self.decode()
        return self.__frame

    def get_timestamp(self) -> timedelta:
        return self.msg.getTimestampDevice(dai.CameraExposureOffset.MIDDLE)

    def get_sequence_num(self) -> int:
        return self.msg.getSequenceNum()

    def set_decode_codec(self, get_codec: Callable):
        self._get_codec = get_codec

    def decode(self) -> Optional[np.ndarray]:
        if self._get_codec is None:
            return self.msg.getCvFrame() if cv2 else None

        codec = self._get_codec()
        if codec is None:
            return self.msg.getCvFrame() if cv2 else None

        # PyAV decoding support H264, H265, JPEG and Lossless JPEG
        enc_packets = codec.parse(self.msg.getData())
        if len(enc_packets) == 0:
            return None

        frames = codec.decode(enc_packets[-1])
        if not frames:
            return None

        return frames[0].to_ndarray(format='bgr24')

    def get_size(self) -> Tuple[int, int]:
        return self.msg.getWidth(), self.msg.getHeight()


class DisparityPacket(FramePacket):
    def __init__(self,
                 name: str,
                 img: dai.ImgFrame,
                 multiplier: float,
                 disparity_map: Optional[np.ndarray] = None,
                 colorize: StereoColor = None,
                 colormap: int = None,
                 aligned_frame: Optional[dai.ImgFrame] = None,
                 confidence_map: Optional[np.ndarray] = None
                 ):
        """
        disparity_map might be filtered, eg. if WLS filter is enabled
        """
        super().__init__(name=name, msg=img)
        self.aligned_frame = aligned_frame
        self.disparity_map = disparity_map
        self.multiplier = multiplier
        self.colorize = colorize
        self.colormap = colormap

        self.confidence_map = confidence_map
        self.depth_score = None
        if self.confidence_map:
            values = 1 - (self.confidence_map.getData() / 255)
            values_no_outliers = values[np.logical_and(values > 0.0, values < 1.0)]
            self.depth_score = np.mean(values_no_outliers)

    def get_disparity(self) -> np.ndarray:
        if self.disparity_map is not None:
            return self.disparity_map
        else:
            self.msg.getFrame()

    def get_colorized_frame(self, visualizer) -> np.ndarray:
        frame = self.get_disparity()
        colorized_disp = frame * self.multiplier

        try:
            aligned_frame = self.aligned_frame.getCvFrame()
        except AttributeError:
            aligned_frame = None

        stereo_config = visualizer.config.stereo

        colorize = self.colorize or stereo_config.colorize
        if self.colormap is not None:
            colormap = self.colormap
        else:
            colormap = stereo_config.colormap
            colormap[0] = [0, 0, 0]  # Invalidate pixels 0 to be black

        if aligned_frame is not None and colorized_disp.ndim == 2 and aligned_frame.ndim == 3:
            colorized_disp = colorized_disp[..., np.newaxis]

        if colorize == StereoColor.GRAY:
            pass
        elif colorize == StereoColor.RGB:
            colorized_disp = cv2.applyColorMap(colorized_disp.astype(np.uint8), colormap)
        elif colorize == StereoColor.RGBD:
            colorized_disp = cv2.applyColorMap(
                (colorized_disp + aligned_frame * 0.5).astype(np.uint8), colormap
            )
        return colorized_disp


class DepthPacket(FramePacket):
    def __init__(self, name: str,
                 msg: dai.ImgFrame):
        super().__init__(name, msg)
        self.depth = msg.getFrame()


class DisparityDepthPacket(DisparityPacket):
    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 colorize: StereoColor = None,
                 colormap: int = None,
                 aligned_frame: Optional[dai.ImgFrame] = None,
                 disp_scale_factor=255 / 95,
                 confidence_map=None
                 ):
        # DepthPacket.__init__(self, name=name, msg=img_frame)
        super().__init__(
            name=name,
            img=img_frame,
            disparity_map=None,
            multiplier=255 / 95,
            colorize=colorize,
            colormap=colormap,
            aligned_frame=aligned_frame,
            confidence_map=confidence_map
        )
        self.disp_scale_factor = disp_scale_factor

    def get_disparity(self) -> np.ndarray:
        with np.errstate(divide='ignore'):
            disparity = self.disp_scale_factor / self.msg.getFrame()
        disparity[disparity == np.inf] = 0
        return disparity

    # def get_colorized_frame(self, visualizer) -> np.ndarray:
    # Convert depth to disparity for nicer visualization


class PointcloudPacket(BasePacket):
    def __init__(self,
                 name: str,
                 points: np.ndarray,
                 depth_map: dai.ImgFrame,
                 colorize_frame: Optional[dai.ImgFrame]):
        super().__init__(name=name)
        self.points = points
        self.colorize_frame = colorize_frame.getCvFrame() if colorize_frame is not None else None
        self.depth_map = depth_map

    def get_sequence_num(self) -> int:
        return self.depth_map.getSequenceNum()

    def get_timestamp(self) -> timedelta:
        return self.depth_map.getTimestampDevice()

    def crop_points(self, bb: BoundingBox) -> np.ndarray:
        """
        Crop points to the bounding box

        Returns: Cropped section of the point cloud
        """
        x1, y1, x2, y2 = bb.to_tuple(self.points.shape)
        return self.points[y1:y2, x1:x2]


class SpatialBbMappingPacket(DisparityDepthPacket):
    """
    Output from Spatial Detection nodes - depth frame + bounding box mappings. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 spatials: dai.SpatialImgDetections,
                 disp_scale_factor: float):
        super().__init__(name=name,
                         img_frame=msg,
                         disp_scale_factor=disp_scale_factor)
        self.spatials = spatials

    def prepare_visualizer_objects(self, vis: Visualizer) -> None:
        # Add detections to packet
        for detection in self.spatials.detections:
            br = detection.boundingBoxMapping.roi.bottomRight()
            tl = detection.boundingBoxMapping.roi.topLeft()
            bbox = BoundingBox([tl.x, tl.y, br.x, br.y])
            # Add detections to visualizer
            vis.add_bbox(
                bbox=bbox,
                thickness=3,
                color=(0, 0, 0)
            )
            vis.add_bbox(
                bbox=bbox,
                thickness=1,
                color=(255, 255, 255)
            )


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
        for detection in self.detections:
            # Add detections to visualizer
            vis.add_bbox(
                bbox=self.bbox.get_relative_bbox(detection.bbox),
                # label=detection.label_str,
                color=detection.color,
            )
            vis.add_text(
                f'{detection.label_str} {100 * detection.confidence:.0f}%',
                bbox=self.bbox.get_relative_bbox(detection.bbox),
                position=TextPosition.TOP_LEFT,
            )
            # Spatial coordinates
            if type(detection.img_detection) == dai.SpatialImgDetection:
                x_meters = detection.img_detection.spatialCoordinates.x / 1000
                y_meters = detection.img_detection.spatialCoordinates.y / 1000
                z_meters = detection.img_detection.spatialCoordinates.z / 1000
                vis.add_text(
                    f'X: {x_meters:.2f}m\nY: {y_meters:.2f}m\nZ: {z_meters:.2f}m',
                    bbox=self.bbox.get_relative_bbox(detection.bbox),
                    position=TextPosition.BOTTOM_LEFT,
                )


class TrackerPacket(FramePacket):
    """
    Output of Object Tracker node. Tracklets + Image frame. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 tracklets: dai.Tracklets,
                 bbox: BoundingBox,
                 ):
        super().__init__(name=name,
                         msg=msg)

        # int: object_id, list: TrackingDetection
        self.tracklets: Dict[int, List[TrackingDetection]] = {}
        self.daiTracklets = tracklets
        self.bbox = bbox

    def _is_spatial_detection(self) -> bool:
        coords = self.daiTracklets.tracklets[0].spatialCoordinates
        return coords.x != 0.0 or coords.y != 0.0 or coords.z != 0.0

    def prepare_visualizer_objects(self, visualizer: Visualizer) -> None:
        tracking_config = visualizer.config.tracking
        for obj_id, tracking_dets in self.tracklets.items():
            tracking_det = tracking_dets[-1]  # Get the last detection
            bb = tracking_det.filtered_2d or tracking_det.bbox
            visualizer.add_bbox(
                bbox=self.bbox.get_relative_bbox(bb),
                label=f"[{obj_id}] {tracking_det.label_str}",
                color=tracking_det.color,
            )
            visualizer.add_text(
                f'{tracking_det.label_str} {100 * tracking_det.confidence:.0f}%',
                bbox=self.bbox.get_relative_bbox(bb),
                position=TextPosition.TOP_LEFT,
            )
            if visualizer.config.tracking.show_speed and \
                    tracking_det.speed is not None:
                visualizer.add_text(
                    text=f"{tracking_det.speed:.2f} m/s",
                    color=tracking_det.color,
                    bbox=self.bbox.get_relative_bbox(bb),
                    position=TextPosition.BOTTOM_RIGHT,
                )
            w, h = self.get_size()
            tracklet_length = 0
            for i in reversed(range(len(tracking_dets) - 1)):
                p1 = self.bbox.get_relative_bbox(tracking_dets[i].bbox).get_centroid().denormalize((h, w))
                p2 = self.bbox.get_relative_bbox(tracking_dets[i + 1].bbox).get_centroid().denormalize((h, w))

                if tracking_config.max_length != -1:
                    tracklet_length += np.linalg.norm(np.array(p1) - np.array(p2))
                    if tracking_config.max_length < tracklet_length:
                        break

                thickness = tracking_config.line_thickness
                if tracking_config.fading_tails:
                    thickness = max(1, int(np.ceil(thickness * i / len(tracking_dets))))

                visualizer.add_line(pt1=p1, pt2=p2,
                                    color=tracking_dets[i].color,
                                    thickness=thickness
                                    )


class TwoStagePacket(DetectionPacket):
    """
    Output of 2-stage NN pipeline; Image frame, Image detections and multiple NNData results. Inherits DetectionPacket.
    """

    def __init__(self, name: str,
                 msg: dai.ImgFrame,
                 img_detections: dai.ImgDetections,
                 nn_data: List[dai.NNData],
                 labels: List[int],
                 bbox: BoundingBox
                 ):
        super().__init__(name=name,
                         msg=msg,
                         dai_msg=img_detections,
                         bbox=bbox
                         )
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
    def __init__(self, name, packet: dai.IMUPacket, rotation=None):
        self.packet = packet
        super().__init__(name)

        self.acceleroMeter = packet.acceleroMeter
        self.gyroscope = packet.gyroscope
        self.magneticField = packet.magneticField
        self.rotationVector = rotation if rotation is not None else packet.rotationVector

        # Check which reports are available
        self.available_reports: Dict[str, dai.IMUReport] = {}
        for i, val in enumerate([self.acceleroMeter, self.gyroscope, self.magneticField, self.rotationVector]):
            if (i == 3 and rotation) or val.getTimestampDevice() != timedelta(0):
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

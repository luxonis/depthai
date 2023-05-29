from typing import Tuple, List, Union, Optional
from depthai_sdk.components.hand_tracker.mediapipe_utils import HandRegion, Body
import depthai as dai
import numpy as np
import math
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

class NNDataPacket:
    """
    Contains only dai.NNData message
    """
    name: str  # NNData stream name
    msg: dai.NNData  # Original depthai message

    def __init__(self, name: str, nn_data: dai.NNData):
        self.name = name
        self.msg = nn_data

class FramePacket:
    """
    Contains only dai.ImgFrame message and cv2 frame, which is used by visualization logic.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 frame: Optional[np.ndarray],
                 visualizer: 'Visualizer' = None):
        self.name = name
        self.msg = msg
        self.frame = frame

        self.visualizer = visualizer


class PointcloudPacket:
    def __init__(self,
                 name: str,
                 points: np.ndarray,
                 depth_map: dai.ImgFrame,
                 color_frame: Optional[np.ndarray] = None,
                 visualizer: 'Visualizer' = None):
        self.name = name
        self.points = points
        self.depth_imgFrame = depth_map
        self.color_frame = color_frame
        self.visualizer = visualizer

class HandTrackerPacket:
    def __init__(self,
                 name: str,
                 hands: List[HandRegion],
                 color_frame: Optional[np.ndarray] = None,
                 visualizer: 'Visualizer' = None):
        self.name = name
        self.hands = hands
        self.color_frame = color_frame
        self.visualizer = visualizer

class BodyPosePacket:
    def __init__(self,
                 name: str,
                 body: Body,
                 color_frame: Optional[np.ndarray] = None,
                 visualizer: 'Visualizer' = None):
        self.name = name
        self.body = body
        self.color_frame = color_frame
        self.visualizer = visualizer

class DepthPacket(FramePacket):
    mono_frame: dai.ImgFrame

    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 mono_frame: Optional[dai.ImgFrame],
                 depth_map: Optional[np.ndarray] = None,
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         msg=img_frame,
                         frame=img_frame.getCvFrame() if cv2 else None,
                         visualizer=visualizer)

        if mono_frame is not None:
            self.mono_frame = mono_frame

        self.depth_map = depth_map

class SpatialBbMappingPacket(FramePacket):
    """
    Output from Spatial Detection nodes - depth frame + bounding box mappings. Inherits FramePacket.
    """
    spatials: dai.SpatialImgDetections

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 spatials: dai.SpatialImgDetections,
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         msg=msg,
                         frame=msg.getFrame() if cv2 else None,
                         visualizer=visualizer)
        self.spatials = spatials


class RotatedDetectionPacket(FramePacket):
    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 rotated_rects: List[dai.RotatedRect],
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         msg=msg,
                         frame=msg.getCvFrame() if cv2 else None,
                         visualizer=visualizer)
        self.rotated_rects = rotated_rects
        self.bb_corners = [self.rotated_rect_to_points(rr) for rr in self.rotated_rects]

    def rotated_rect_to_points(self, rr: dai.RotatedRect) -> List[Tuple]:
        cx = rr.center.x
        cy = rr.center.y
        w = rr.size.width / 2  # half-width
        h = rr.size.height / 2  # half-height
        rotation = math.radians(rr.angle)  # convert angle to radians

        b = math.cos(rotation)
        a = math.sin(rotation)

        # calculate corners
        p0x = cx - a*h - b*w
        p0y = cy + b*h - a*w
        p1x = cx + a*h - b*w
        p1y = cy - b*h - a*w
        p2x = 2*cx - p0x
        p2y = 2*cy - p0y
        p3x = 2*cx - p1x
        p3y = 2*cy - p1y

        return [(p0x,p0y), (p1x,p1y), (p2x,p2y), (p3x,p3y)]

class DetectionPacket(FramePacket):
    """
    Output from Detection Network nodes - image frame + image detections. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 img_detections: Union[dai.ImgDetections, dai.SpatialImgDetections],
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         msg=msg,
                         frame=msg.getCvFrame() if cv2 else None,
                         visualizer=visualizer)
        self.img_detections = img_detections
        self.detections = []

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


class TrackerPacket(FramePacket):
    """
    Output of Object Tracker node. Tracklets + Image frame. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 msg: dai.ImgFrame,
                 tracklets: dai.Tracklets,
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         msg=msg,
                         frame=msg.getCvFrame() if cv2 else None,
                         visualizer=visualizer)
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
                 labels: List[int],
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         msg=msg,
                         img_detections=img_detections,
                         visualizer=visualizer)
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


class IMUPacket:
    def __init__(self, data: List[dai.IMUData]):
        self.data = data

    def __str__(self):
        packet_details = []

        for imu_data in self.data:
            # TODO print more details if needed
            accelerometer_str = 'Accelerometer [m/s^2]: (x: %.2f, y: %.2f, z: %.2f)' % (
                imu_data.acceleroMeter.x,
                imu_data.acceleroMeter.y,
                imu_data.acceleroMeter.z
            )

            gyroscope_str = 'Gyroscope [rad/s]: (x: %.2f, y: %.2f, z: %.2f)' % (
                imu_data.gyroscope.x,
                imu_data.gyroscope.y,
                imu_data.gyroscope.z
            )

            packet_details.append(f'{accelerometer_str}, {gyroscope_str})')

        return f'IMU Packet: {packet_details}'

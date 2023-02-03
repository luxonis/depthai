from typing import Tuple, List, Union, Optional

import depthai as dai
import numpy as np

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


class _TwoStageDetection(_Detection):
    nn_data: dai.NNData


class FramePacket:
    """
    Contains only dai.ImgFrame message and cv2 frame, which is used by visualization logic.
    """

    name: str  # ImgFrame stream name
    imgFrame: dai.ImgFrame  # Original depthai message
    frame: Optional[np.ndarray]  # cv2 frame for visualization

    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 frame: Optional[np.ndarray],
                 visualizer: 'Visualizer' = None):
        self.name = name
        self.imgFrame = img_frame
        self.frame = frame
        self.visualizer = visualizer


class DepthPacket(FramePacket):
    mono_frame: dai.ImgFrame

    def __init__(self,
                 name: str,
                 disparity_frame: dai.ImgFrame,
                 mono_frame: dai.ImgFrame,
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         img_frame=disparity_frame,
                         frame=disparity_frame.getCvFrame() if cv2 else None,
                         visualizer=visualizer)
        self.mono_frame = mono_frame


class SpatialBbMappingPacket(FramePacket):
    """
    Output from Spatial Detection nodes - depth frame + bounding box mappings. Inherits FramePacket.
    """
    spatials: dai.SpatialImgDetections

    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 spatials: dai.SpatialImgDetections,
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         img_frame=img_frame,
                         frame=img_frame.getFrame() if cv2 else None,
                         visualizer=visualizer)
        self.spatials = spatials


class DetectionPacket(FramePacket):
    """
    Output from Detection Network nodes - image frame + image detections. Inherits FramePacket.
    """

    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 img_detections: Union[dai.ImgDetections, dai.SpatialImgDetections],
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         img_frame=img_frame,
                         frame=img_frame.getCvFrame() if cv2 else None,
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
                 img_frame: dai.ImgFrame,
                 tracklets: dai.Tracklets,
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         img_frame=img_frame,
                         frame=img_frame.getCvFrame() if cv2 else None,
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
                 img_frame: dai.ImgFrame,
                 img_detections: dai.ImgDetections,
                 nn_data: List[dai.NNData],
                 labels: List[int],
                 visualizer: 'Visualizer' = None):
        super().__init__(name=name,
                         img_frame=img_frame,
                         img_detections=img_detections,
                         visualizer=visualizer)
        self.frame = self.imgFrame.getCvFrame() if cv2 else None
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

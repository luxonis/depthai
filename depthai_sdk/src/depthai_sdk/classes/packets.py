from typing import Tuple, List, Union

import depthai as dai
import numpy as np


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
    frame: np.ndarray  # cv2 frame for visualization

    def __init__(self, name: str, imgFrame: dai.ImgFrame, frame: np.ndarray):
        self.name = name
        self.imgFrame = imgFrame
        self.frame = frame


class DepthPacket(FramePacket):
    mono_frame: dai.ImgFrame

    def __init__(self, name: str, disparity_frame: dai.ImgFrame, mono_frame: dai.ImgFrame):
        super().__init__(name, disparity_frame, disparity_frame.getCvFrame())
        self.mono_frame = mono_frame


class SpatialBbMappingPacket(FramePacket):
    """
    Output from Spatial Detection nodes - depth frame + bounding box mappings. Inherits FramePacket.
    """
    spatials: dai.SpatialImgDetections

    def __init__(self, name: str, img_frame: dai.ImgFrame, spatials: dai.SpatialImgDetections):
        super().__init__(name, img_frame, img_frame.getFrame())
        self.spatials = spatials


class DetectionPacket(FramePacket):
    """
    Output from Detection Network nodes - image frame + image detections. Inherits FramePacket.
    """
    img_detections: Union[dai.ImgDetections, dai.SpatialImgDetections]
    detections: List[_Detection]

    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 img_detections: Union[dai.ImgDetections, dai.SpatialImgDetections]):
        super().__init__(name, img_frame, img_frame.getCvFrame())
        self.img_detections = img_detections
        self.detections = []

    def _is_spatial_detection(self) -> bool:
        return isinstance(self.img_detections, dai.SpatialImgDetections)

    def _add_detection(self, img_det: dai.ImgDetection, bbox: np.ndarray, txt: str, color):
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
    daiTracklets: dai.Tracklets
    detections: List[_TrackingDetection]

    def __init__(self,
                 name: str,
                 img_frame: dai.ImgFrame,
                 tracklets: dai.Tracklets):
        super().__init__(name, img_frame, img_frame.getCvFrame())
        self.daiTracklets = tracklets
        self.detections = []

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
    nnData: List[dai.NNData]
    labels: List[int] = None
    _cntr: int = 0  # Label counter

    def __init__(self, name: str,
                 img_frame: dai.ImgFrame,
                 img_detections: dai.ImgDetections,
                 nn_data: List[dai.NNData],
                 labels: List[int]):
        super().__init__(name, img_frame, img_detections)
        self.frame = self.imgFrame.getCvFrame()
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
    data: List[dai.IMUData]

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

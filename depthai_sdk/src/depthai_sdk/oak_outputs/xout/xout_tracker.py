import logging
import math
from collections import defaultdict
from typing import Union, Dict, Optional

import depthai as dai
import numpy as np
from depthai_sdk.classes import DetectionPacket, TrackerPacket
from depthai_sdk.classes.packets import _TrackingDetection
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_nn import XoutNnResults
from depthai_sdk.tracking import KalmanFilter
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.configs import TextPosition
from depthai_sdk.visualize.visualizer import Visualizer


class XoutTracker(XoutNnResults):
    buffer_size: int = 10

    def __init__(self,
                 det_nn: 'NNComponent',
                 frames: StreamXout,
                 device: dai.Device,
                 tracklets: StreamXout,
                 apply_kalman: bool = False,
                 forget_after_n_frames: Optional[int] = None,
                 calculate_speed: bool = False):
        super().__init__(det_nn, frames, tracklets)
        self.name = 'Object Tracker'
        self.device = device

        self.__read_device_calibration()

        self.buffer = []
        self.spatial_buffer = []

        self.lost_counter = {}
        self.blacklist = set()

        self.apply_kalman = apply_kalman
        self.forget_after_n_frames = forget_after_n_frames
        self.kalman_filters: Dict[int, Dict[str, KalmanFilter]] = {}
        self.calculate_speed = calculate_speed

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        super().setup_visualize(visualizer, visualizer_enabled, name)

    def on_callback(self, packet: Union[DetectionPacket, TrackerPacket]):
        if len(packet.frame.shape) == 2:
            packet.frame = np.dstack((packet.frame, packet.frame, packet.frame))

        frame_shape = self.det_nn._input.stream_size[::-1]

        if self._frame_shape is None:
            # Lazy-load the frame shape
            self._frame_shape = np.array([*frame_shape])
            if self._visualizer:
                self._visualizer.frame_shape = self._frame_shape

        spatial_points = self._get_spatial_points(packet)
        threshold = self.forget_after_n_frames

        if threshold:
            self._update_lost_counter(packet, threshold)

        self._update_buffers(packet, spatial_points)

        # Optional kalman filter
        if self.apply_kalman:
            self._kalman_filter(packet, spatial_points)

        # Estimate speed
        tracklet2speed = self._calculate_speed(spatial_points)

        if self._visualizer:
            self._add_tracklet_visualization(packet, spatial_points, tracklet2speed)

        self._add_detections(packet, tracklet2speed)

    def visualize(self, packet):
        super().visualize(packet)

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

    def _add_tracklet_visualization(self, packet, spatial_points, tracklet2speed):
        h, w = self._frame_shape[:2]
        filtered_tracklets = [tracklet for tracklet in packet.daiTracklets.tracklets if
                              tracklet.id not in self.blacklist]

        norm_bbox = BoundingBox().resize_to_aspect_ratio(packet.frame.shape, self._nn_size, self._resize_mode)

        self._visualizer.add_detections(detections=filtered_tracklets,
                                        normalizer=norm_bbox,
                                        label_map=self.labels,
                                        spatial_points=spatial_points)

        # Add tracking ids
        for tracklet in filtered_tracklets:
            det = tracklet.srcImgDetection
            bbox = (w * det.xmin, h * det.ymin, w * det.xmax, h * det.ymax)
            bbox = tuple(map(int, bbox))
            self._visualizer.add_text(
                f'ID: {tracklet.id}',
                bbox=bbox,
                position=TextPosition.MID
            )

            if self._visualizer.config.tracking.show_speed and tracklet.id in tracklet2speed:
                speed = tracklet2speed[tracklet.id]
                speed = f'{speed:.1f} m/s\n{speed * 3.6:.1f} km/h'
                bbox = tracklet.srcImgDetection
                bbox = (int(w * bbox.xmin), int(h * bbox.ymin), int(w * bbox.xmax), int(h * bbox.ymax))

                self._visualizer.add_text(
                    speed,
                    bbox=bbox,
                    position=TextPosition.TOP_RIGHT,
                    outline=True
                )

        # Add tracking lines
        self._visualizer.add_trail(
            tracklets=[t for p in self.buffer for t in p.daiTracklets.tracklets if t.id not in self.blacklist],
            label_map=self.labels,
            bbox=norm_bbox,
        )

    def _update_lost_counter(self, packet, lost_threshold: int):
        for i, tracklet in enumerate(packet.daiTracklets.tracklets):
            if tracklet.status == dai.Tracklet.TrackingStatus.NEW:
                self.__remove_from_blacklist(tracklet)
                self.lost_counter[tracklet.id] = 0
            elif tracklet.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.__remove_from_blacklist(tracklet)
                self.lost_counter[tracklet.id] = 0
            elif tracklet.status == dai.Tracklet.TrackingStatus.LOST and tracklet.id in self.lost_counter:
                self.lost_counter[tracklet.id] += 1

            if tracklet.id in self.lost_counter and self.lost_counter[tracklet.id] >= lost_threshold:
                self.__add_to_blacklist(tracklet)
                self.lost_counter.pop(tracklet.id)

    def _update_buffers(self, packet, spatial_points=None):
        # Update buffer
        self.buffer.append(packet)
        if self.buffer_size < len(self.buffer):
            self.buffer.pop(0)

        # Update spatial buffer
        if spatial_points is not None:
            self.spatial_buffer.append(spatial_points)
            if self.buffer_size < 5:
                self.spatial_buffer.pop(0)

    def _kalman_filter(self, packet, spatial_points=None):
        current_time = packet.daiTracklets.getTimestamp()
        is_3d = spatial_points is not None

        tracklets = []

        for i, tracklet in enumerate(packet.daiTracklets.tracklets):
            if tracklet.id in self.blacklist:  # Skip blacklisted tracklets
                continue

            meas_vec_space = 0
            meas_std_space = 0

            roi = tracklet.roi
            x1 = roi.topLeft().x
            y1 = roi.topLeft().y
            x2 = roi.bottomRight().x
            y2 = roi.bottomRight().y

            if is_3d:
                x_space = tracklet.spatialCoordinates.x
                y_space = tracklet.spatialCoordinates.y
                z_space = tracklet.spatialCoordinates.z
                meas_vec_space = np.array([[x_space], [y_space], [z_space]])
                meas_std_space = z_space ** 2 / (self.baseline * self.focal)

            meas_vec_bbox = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]])

            if tracklet.status == dai.Tracklet.TrackingStatus.NEW:
                self.kalman_filters[tracklet.id] = {'bbox': KalmanFilter(10, 0.1, meas_vec_bbox, current_time)}
                if is_3d:
                    self.kalman_filters[tracklet.id]['space'] = KalmanFilter(10, 0.1, meas_vec_space, current_time)

            elif tracklet.status == dai.Tracklet.TrackingStatus.TRACKED or tracklet.status == dai.Tracklet.TrackingStatus.LOST:
                if tracklet.id not in self.kalman_filters:
                    continue

                dt = current_time - self.kalman_filters[tracklet.id]['bbox'].time
                dt = dt.total_seconds()

                self.kalman_filters[tracklet.id]['bbox'].predict(dt)
                self.kalman_filters[tracklet.id]['bbox'].update(meas_vec_bbox)
                self.kalman_filters[tracklet.id]['bbox'].time = current_time
                vec_bbox = self.kalman_filters[tracklet.id]['bbox'].x

                if is_3d:
                    self.kalman_filters[tracklet.id]['space'].predict(dt)
                    self.kalman_filters[tracklet.id]['space'].update(meas_vec_space)
                    self.kalman_filters[tracklet.id]['space'].time = current_time
                    self.kalman_filters[tracklet.id]['space'].meas_std = meas_std_space
                    vec_space = self.kalman_filters[tracklet.id]['space'].x

                x1_filter = vec_bbox[0] - vec_bbox[2] / 2
                x2_filter = vec_bbox[0] + vec_bbox[2] / 2
                y1_filter = vec_bbox[1] - vec_bbox[3] / 2
                y2_filter = vec_bbox[1] + vec_bbox[3] / 2

                rect = dai.Rect(x1_filter, y1_filter, x2_filter - x1_filter, y2_filter - y1_filter)
                new_tracklet = self.__create_tracklet(tracklet, rect, vec_space if is_3d else None)
                tracklets.append(new_tracklet)

            elif tracklet.status == dai.Tracklet.TrackingStatus.REMOVED:
                self.kalman_filters.pop(tracklet.id, None)

            if tracklets:
                packet.daiTracklets.tracklets = tracklets

    def _add_detections(self, packet, tracklet2speed):
        for tracklet in packet.daiTracklets.tracklets:
            if tracklet.id in self.blacklist:  # Skip blacklisted tracklets
                continue

            d = _TrackingDetection()
            img_d = tracklet.srcImgDetection
            d.tracklet = tracklet
            d.label = self.labels[img_d.label][0] if self.labels else str(img_d.label)
            d.color = self.labels[img_d.label][1] if self.labels else (255, 255, 255)
            roi = tracklet.roi.denormalize(self._frame_shape[1], self._frame_shape[0])
            d.top_left = (int(roi.x), int(roi.y))
            d.bottom_right = (int(roi.x + roi.width), int(roi.y + roi.height))

            if tracklet.id in tracklet2speed:
                d.speed = tracklet2speed[tracklet.id]
                d.speed_kmph = d.speed * 3.6
                d.speed_mph = d.speed * 2.23694

            packet.detections.append(d)

    def _calculate_speed(self, spatial_points) -> dict:
        if spatial_points is None or self.calculate_speed is False:
            return {}

        tracklet2speed = {}
        if spatial_points is not None:
            spatial_coords = defaultdict(list)
            t = defaultdict(list)
            tracklets = defaultdict(list)
            for buffered_packet in self.buffer:
                for tracklet in buffered_packet.daiTracklets.tracklets:
                    spatial_coords[tracklet.id].append(tracklet.spatialCoordinates)
                    t[tracklet.id].append(buffered_packet.daiTracklets.getTimestamp())
                    tracklets[tracklet.id].append(tracklet)

            indices = spatial_coords.keys()
            for idx in indices:
                # Skip if there is only one point
                if len(spatial_coords[idx]) < 2:
                    continue

                n = len(spatial_coords[idx])
                speeds = []

                for i in range(n - 1):
                    x1, y1, z1 = spatial_coords[idx][i].x, spatial_coords[idx][i].y, spatial_coords[idx][i].z
                    x2, y2, z2 = spatial_coords[idx][i + 1].x, spatial_coords[idx][i + 1].y, spatial_coords[idx][
                        i + 1].z
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) / 1000
                    time = (t[idx][i + 1] - t[idx][i]).total_seconds()
                    speeds.append(distance / time)

                window_size = 3
                window = np.hanning(window_size)
                window /= window.sum()

                smoothed = np.convolve(speeds, window, mode='same')
                speed = np.mean(smoothed)

                tracklet2speed[idx] = speed

        return tracklet2speed

    @staticmethod
    def _get_spatial_points(packet) -> list:
        try:
            if packet._is_spatial_detection():
                spatial_points = [packet._get_spatials(det.srcImgDetection)
                                  for det in
                                  packet.daiTracklets.tracklets]
            else:
                spatial_points = None
        except IndexError:
            spatial_points = None

        return spatial_points

    def __get_img_detection(self, tracklet, confidence: float = 1.0):
        """Converts tracklet to ImgDetection."""
        img_d = dai.ImgDetection()
        img_d.label = tracklet.label
        img_d.confidence = confidence
        img_d.xmin = tracklet.roi.x
        img_d.ymin = tracklet.roi.y
        img_d.xmax = tracklet.roi.x + tracklet.roi.width
        img_d.ymax = tracklet.roi.y + tracklet.roi.height
        return img_d

    def __create_tracklet(self, tracklet, roi=None, spatial_points=None):
        """Creates a Tracklet object."""
        tracklet_obj = dai.Tracklet()
        tracklet_obj.id = tracklet.id
        tracklet_obj.age = tracklet.age
        tracklet_obj.label = tracklet.label
        tracklet_obj.status = tracklet.status
        tracklet_obj.roi = roi
        if spatial_points is not None:
            tracklet_obj.spatialCoordinates = dai.Point3f(spatial_points[0], spatial_points[1], spatial_points[2])
        else:
            tracklet_obj.spatialCoordinates = tracklet.spatialCoordinates

        img_d = self.__get_img_detection(tracklet, confidence=tracklet.srcImgDetection.confidence)
        tracklet_obj.srcImgDetection = img_d
        return tracklet_obj

    def __read_device_calibration(self):
        calib = self.device.readCalibration()
        eeprom = calib.getEepromData()
        left_cam = calib.getStereoLeftCameraId()
        if left_cam != dai.CameraBoardSocket.AUTO and left_cam in eeprom.cameraData.keys():
            cam_info = eeprom.cameraData[left_cam]
            self.baseline = abs(cam_info.extrinsics.specTranslation.x * 10)  # cm -> mm
            fov = calib.getFov(calib.getStereoLeftCameraId())
            self.focal = (cam_info.width / 2) / (2. * math.tan(math.radians(fov / 2)))
        else:
            logging.warning("Calibration data missing, using OAK-D defaults")
            self.baseline = 75
            self.focal = 440

    def __add_to_blacklist(self, tracklet):
        if tracklet.id not in self.blacklist:
            self.blacklist.add(tracklet.id)

    def __remove_from_blacklist(self, tracklet):
        if tracklet.id in self.blacklist:
            self.blacklist.remove(tracklet.id)

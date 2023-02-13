from collections import defaultdict
from typing import Union, Dict

import depthai as dai
import numpy as np

from depthai_sdk.classes import DetectionPacket, TrackerPacket
from depthai_sdk.classes.packets import _TrackingDetection
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_nn import XoutNnResults
from depthai_sdk.visualize.configs import TextPosition
from depthai_sdk.visualize.visualizer import Visualizer


class XoutTracker(XoutNnResults):
    buffer_size: int = 10

    def __init__(self, det_nn, frames: StreamXout, tracklets: StreamXout):
        super().__init__(det_nn, frames, tracklets)
        self.buffer = []
        self.spatial_buffer = []
        self.name = 'Object Tracker'
        self.lost_counter = {}

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None):
        super().setup_visualize(visualizer, visualizer_enabled, name)

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

        if self._visualizer and self._visualizer.frame_shape is None:
            if packet.frame.ndim == 1:
                self._visualizer.frame_shape = self._frame_shape
            else:
                self._visualizer.frame_shape = packet.frame.shape

        blacklist = set()
        threshold = 5  # TODO make not visualizer specific (old code: self._visualizer.config.tracking.deletion_lost_threshold)
        for i, tracklet in enumerate(packet.daiTracklets.tracklets):
            if tracklet.status == dai.Tracklet.TrackingStatus.NEW:
                self.lost_counter[tracklet.id] = 0
            elif tracklet.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.lost_counter[tracklet.id] = 0
            elif tracklet.status == dai.Tracklet.TrackingStatus.LOST and tracklet.id in self.lost_counter:
                self.lost_counter[tracklet.id] += 1

            if tracklet.id in self.lost_counter and self.lost_counter[tracklet.id] >= threshold:
                blacklist.add(tracklet.id)
                self.lost_counter.pop(tracklet.id)

        h, w = packet.frame.shape[:2]

        # Update buffer
        self.buffer.append(packet)
        if self.buffer_size < len(self.buffer):
            self.buffer.pop(0)

        # Update spatial buffer
        if spatial_points is not None:
            self.spatial_buffer.append(spatial_points)
            if self.buffer_size < 5:
                self.spatial_buffer.pop(0)

        # Estimate speed
        tracklet2speed = self._calculate_speed(spatial_points)

        if self._visualizer:
            filtered_tracklets = [tracklet for tracklet in packet.daiTracklets.tracklets if
                                  tracklet.id not in blacklist]
            self._visualizer.add_detections(filtered_tracklets,
                                            self.normalizer,
                                            self.labels,
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

                if self._visualizer.config.tracking.speed and tracklet.id in tracklet2speed:
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
                tracklets=[t for p in self.buffer for t in p.daiTracklets.tracklets if t.id not in blacklist],
                label_map=self.labels
            )

        for tracklet in packet.daiTracklets.tracklets:
            d = _TrackingDetection()
            detection = tracklet.srcImgDetection
            d.img_detection = detection
            d.tracklet = tracklet
            d.label = self.labels[detection.label][0] if self.labels else str(detection.label)
            d.color = self.labels[detection.label][1] if self.labels else (255, 255, 255)
            bbox = self.normalizer.normalize(
                frame=np.zeros(self._frame_shape, dtype=bool),
                bbox=(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            d.top_left = (int(bbox[0]), int(bbox[1]))
            d.bottom_right = (int(bbox[2]), int(bbox[3]))

            if tracklet.id in tracklet2speed:
                d.speed = tracklet2speed[tracklet.id]
                d.speed_kmph = d.speed * 3.6
                d.speed_mph = d.speed * 2.23694

            packet.detections.append(d)

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

    def _calculate_speed(self, spatial_points) -> dict:
        if spatial_points is None:
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

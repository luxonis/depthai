from collections import defaultdict
from typing import Union, Dict

import depthai as dai
import numpy as np

from depthai_sdk.classes import DetectionPacket, TrackerPacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_nn import XoutNnResults
from depthai_sdk.visualize.configs import TextPosition


class XoutTracker(XoutNnResults):
    buffer_size: int = 10

    def __init__(self, det_nn, frames: StreamXout, tracklets: StreamXout):
        super().__init__(det_nn, frames, tracklets)
        self.buffer = []
        self.spatial_buffer = []
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
            if tracklet.status == dai.Tracklet.TrackingStatus.NEW:
                self.lost_counter[tracklet.id] = 0
            elif tracklet.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.lost_counter[tracklet.id] = 0
            elif tracklet.status == dai.Tracklet.TrackingStatus.LOST and tracklet.id in self.lost_counter:
                self.lost_counter[tracklet.id] += 1

            if tracklet.id in self.lost_counter and self.lost_counter[tracklet.id] >= threshold:
                blacklist.add(tracklet.id)
                self.lost_counter.pop(tracklet.id)

        filtered_tracklets = [tracklet for tracklet in packet.daiTracklets.tracklets if tracklet.id not in blacklist]
        self._visualizer.add_detections(filtered_tracklets,
                                        self.normalizer,
                                        self.labels,
                                        spatial_points=spatial_points)

        # Update buffer
        self.buffer.append(packet)
        if self.buffer_size < len(self.buffer):
            self.buffer.pop(0)

        # Update spatial buffer
        if spatial_points is not None:
            self.spatial_buffer.append(spatial_points)
            if self.buffer_size < 5:
                self.spatial_buffer.pop(0)

        # Add tracking lines
        self._visualizer.add_trail(
            tracklets=[t for p in self.buffer for t in p.daiTracklets.tracklets if t.id not in blacklist],
            label_map=self.labels
        )

        # Add tracking ids
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

        # Estimate speed
        if spatial_points is not None and self._visualizer.config.tracking.speed:
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
                    x1, y1, z1 = x1 / 1000, y1 / 1000, z1 / 1000
                    x2, y2, z2 = x2 / 1000, y2 / 1000, z2 / 1000
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                    time = (t[idx][i + 1] - t[idx][i]).total_seconds()
                    # print(f'Distance: {distance}m, Time: {time}s')
                    speed = distance / time
                    speeds.append(speed)

                window_size = 3
                window = np.hanning(window_size)
                window /= window.sum()
                smoothed = np.convolve(speeds, window, mode='same')

                speed = np.median(smoothed)

                speed = f'{speed:.1f} m/s\n{speed * 3.6:.1f} km/h'
                bbox = tracklets[idx][-1].srcImgDetection
                bbox = (int(w * bbox.xmin), int(h * bbox.ymin), int(w * bbox.xmax), int(h * bbox.ymax))
                self._visualizer.add_text(
                    speed,
                    bbox=bbox,
                    position=TextPosition.TOP_RIGHT,
                    outline=True
                )

    def package(self, msgs: Dict):
        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full
        packet = TrackerPacket(
            self.get_packet_name(),
            msgs[self.frames.name],
            msgs[self.nn_results.name],
        )
        self.queue.put(packet, block=False)

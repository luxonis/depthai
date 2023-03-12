from typing import Union, Dict

import depthai as dai

from depthai_sdk.classes import DetectionPacket, TrackerPacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_nn import XoutNnResults
from depthai_sdk.visualize.configs import TextPosition


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

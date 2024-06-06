import itertools
import warnings
from collections import defaultdict
from functools import cached_property
from typing import List, Optional, Dict

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import DisparityPacket
from depthai_sdk.logger import LOGGER
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_seq_sync import XoutSeqSync
from depthai_sdk.visualize.configs import StereoColor

try:
    import cv2
except ImportError:
    cv2 = None

IR_FLOOD_LIMIT = 765
IR_FLOOD_STEP = IR_FLOOD_LIMIT / 4
IR_DOT_LIMIT = 1200
IR_DOT_STEP = IR_DOT_LIMIT / 4


class XoutDisparity(XoutSeqSync, XoutFrames):
    def __init__(self,
                 device: dai.Device,
                 frames: StreamXout,
                 disp_factor: float,
                 aligned_frame: Optional[StreamXout],
                 fourcc: str = None,
                 colorize: StereoColor = None,
                 colormap: int = None,
                 wls_config: dict = None,
                 ir_settings: dict = None,
                 confidence_map: StreamXout = None):
        self.aligned_frame = aligned_frame
        self.name = 'Disparity'
        self.multiplier = disp_factor
        self.device = device

        self.confidence_map = confidence_map
        self.fig, self.axes = None, None  # for depth score visualization

        self.colorize = colorize
        self.colormap = colormap

        self.ir_settings = ir_settings
        self._dot_projector_brightness = 0  # [0, 1200]
        self._flood_brightness = 0  # [0, 1500]

        self._metrics_buffer = defaultdict(list)
        self._auto_ir_converged = False
        self._checking_neighbourhood = False
        self._converged_metric_value = None

        # Values that will be tested for function approximation
        self._candidate_pairs = list(itertools.product(np.arange(0, IR_DOT_LIMIT + 1, IR_DOT_STEP),
                                                       np.arange(0, IR_FLOOD_LIMIT + 1, IR_FLOOD_STEP)))
        self._neighbourhood_pairs = []
        self._candidate_idx, self._neighbour_idx = 0, 0
        self._X, self._y = [], []

        # Prefer to use WLS level if set, otherwise use lambda and sigma
        self.use_wls_filter = wls_config['enabled'] if wls_config else False
        if self.use_wls_filter:
            wls_level = wls_config['level']
            LOGGER.debug(
                f'Using WLS level: {wls_level.name} (lambda: {wls_level.value[0]}, sigma: {wls_level.value[1]})'
            )
            self.wls_lambda = wls_level.value[0] or wls_config['lambda']
            self.wls_sigma = wls_level.value[1] or wls_config['sigma']

            try:
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
                self.wls_filter.setLambda(self.wls_lambda)
                self.wls_filter.setSigmaColor(self.wls_sigma)

            except AttributeError:
                warnings.warn(
                    'OpenCV version does not support WLS filter. Disabling WLS filter. '
                    'Make sure you have opencv-contrib-python installed. '
                    'If not, run "pip uninstall opencv-python && pip install opencv-contrib-python -U"'
                )
                self.use_wls_filter = False

        XoutFrames.__init__(self, frames=frames, fourcc=fourcc)
        XoutSeqSync.__init__(self, self.xstreams())

    def on_callback(self, packet) -> None:
        if self.ir_settings['auto_mode']:
            self._auto_ir_search(packet.msg.getFrame())

    def xstreams(self) -> List[StreamXout]:
        streams = [self.frames]
        if self.aligned_frame is not None:
            streams.append(self.aligned_frame)
        if self.confidence_map is not None:
            streams.append(self.confidence_map)

        return streams

    def package(self, msgs: Dict) -> DisparityPacket:
        img_frame = msgs[self.frames.name]
        aligned_frame = msgs[self.aligned_frame.name] if self.aligned_frame else None
        confidence_map = msgs[self.confidence_map.name] if self.confidence_map else None
        # TODO: refactor the mess below
        packet = DisparityPacket(
            self.get_packet_name(),
            img_frame,
            self.multiplier,
            disparity_map=None,
            colorize=self.colorize,
            colormap=self.colormap,
            confidence_map=confidence_map,
            aligned_frame=aligned_frame,
        )
        packet._get_codec = self.get_codec

        if self._fourcc is None:
            disparity_frame = img_frame.getFrame()
        else:
            disparity_frame = packet.decode()
            if disparity_frame is None:
                return None

        if aligned_frame and self.use_wls_filter:
            # Perform WLS filtering
            # If we have wls enabled, it means CV2 is installed
            disparity_frame = self.wls_filter.filter(disparity_frame, aligned_frame.getCvFrame())

        packet.disparity_map = disparity_frame

        return packet

    def _auto_ir_search(self, frame: np.ndarray):
        # Perform neighbourhood search if we got worse metric values
        if self._checking_neighbourhood:
            # Increment the neighbourhood index if we have finished checking the current neighbour pair
            if self._ir_grid_search_iteration(frame, self._neighbourhood_pairs, self._neighbour_idx):
                self._neighbour_idx += 1

        # Check if we have finished checking all candidates, done once on the start up
        elif not self._auto_ir_converged:
            # Increment the candidate index if we have finished checking the current candidate pair
            if self._ir_grid_search_iteration(frame, self._candidate_pairs, self._candidate_idx):
                self._candidate_idx += 1

        # Continuously check the consistency of the metric values, if we are in continuous mode
        elif self._auto_ir_converged and self.ir_settings['continuous_mode']:
            self._check_consistency(frame)

    def _ir_grid_search_iteration(self, frame: np.array, candidate_pairs: list = None, candidate_idx: int = 0):
        fill_rate = np.count_nonzero(frame) / frame.size
        self._metrics_buffer['fill_rate'].append(fill_rate)

        if len(self._metrics_buffer['fill_rate']) < 30:
            return False

        if candidate_idx >= len(candidate_pairs):
            # We have exhausted all candidates
            best_idx = np.argmax(self._y)
            self._converged_metric_value = self._y[best_idx]
            self._dot_projector_brightness, self._flood_brightness = self._X[best_idx]
            self._reset_buffers()
            self._auto_ir_converged = True
            self._checking_neighbourhood = False

            LOGGER.debug(f'Auto IR converged: dot projector - {self._dot_projector_brightness}mA, '
                          f'flood - {self._flood_brightness}mA')
        else:
            self._dot_projector_brightness, self._flood_brightness = candidate_pairs[candidate_idx]

        self._update_ir()

        if self._auto_ir_converged:
            return False

        # Skip first half second of frames to allow for auto exposure to settle down
        fill_rate_avg = np.mean(self._metrics_buffer['fill_rate'][15:])

        self._X.append([self._dot_projector_brightness, self._flood_brightness])
        self._y.append(fill_rate_avg)

        self._metrics_buffer['fill_rate'].clear()
        return True

    def _check_consistency(self, frame):
        fill_rate = np.count_nonzero(frame) / frame.size
        self._metrics_buffer['fill_rate'].append(fill_rate)

        if len(self._metrics_buffer['fill_rate']) < 30:
            return

        fill_rate_avg = np.mean(self._metrics_buffer['fill_rate'])
        self._metrics_buffer['fill_rate'].clear()

        if fill_rate_avg < self._converged_metric_value * 0.85:
            self._auto_ir_converged = False
            self._checking_neighbourhood = True
            self._neighbourhood_pairs = np.unique([
                [np.clip(self._dot_projector_brightness + i, 0, IR_DOT_LIMIT),
                 np.clip(self._flood_brightness + j, 0, IR_FLOOD_LIMIT)]
                for i, j in itertools.product([-IR_DOT_STEP / 2, IR_DOT_STEP / 2],
                                              [-IR_FLOOD_STEP / 2, IR_FLOOD_STEP / 2])
            ], axis=0)
            self._neighbour_idx = 0

    def _update_ir(self):
        self.device.setIrLaserDotProjectorBrightness(self._dot_projector_brightness)
        self.device.setIrFloodLightBrightness(self._flood_brightness)

    def _reset_buffers(self):
        self._X, self._y = [], []
        del self._metrics_buffer['fill_rate']

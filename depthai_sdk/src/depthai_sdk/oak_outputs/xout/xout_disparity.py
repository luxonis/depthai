import itertools
import logging
import warnings
from collections import defaultdict
from typing import List, Optional

import depthai as dai
import numpy as np

from depthai_sdk.classes.packets import DepthPacket
from depthai_sdk.oak_outputs.xout import Clickable
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.visualize.configs import StereoColor

try:
    import cv2
except ImportError:
    cv2 = None


class XoutDisparity(XoutFrames, Clickable):
    def __init__(self,
                 device: dai.Device,
                 frames: StreamXout,
                 disp_factor: float,
                 fps: float,
                 mono_frames: Optional[StreamXout],
                 colorize: StereoColor = None,
                 colormap: int = None,
                 wls_config: dict = None,
                 ir_settings: dict = None):
        self.mono_frames = mono_frames
        self.multiplier = disp_factor
        self.fps = fps
        self.name = 'Disparity'
        self.device = device

        self.colorize = colorize
        self.colormap = colormap
        self.use_wls_filter = wls_config['enabled']

        self.ir_settings = ir_settings
        self._dot_projector_brightness = 0  # [0, 1200]
        self._flood_brightness = 0  # [0, 1500]

        self._metrics_buffer = defaultdict(list)
        self._auto_ir_converged = False
        self._checking_neighbourhood = False
        self._converged_metric_value = None

        # Values that will be tested for function approximation
        self._candidate_pairs = list(itertools.product(np.arange(0, 1201, 1200 / 4), np.arange(0, 1501, 1500 / 4)))
        self._neighbourhood_pairs = []
        self._candidate_idx, self._neighbour_idx = 0, 0
        self._X, self._y = [], []

        # Prefer to use WLS level if set, otherwise use lambda and sigma
        wls_level = wls_config['level']
        if wls_level and self.use_wls_filter:
            logging.debug(
                f'Using WLS level: {wls_level.name} (lambda: {wls_level.value[0]}, sigma: {wls_level.value[1]})'
            )
            self.wls_lambda = wls_level.value[0]
            self.wls_sigma = wls_level.value[1]
        else:
            self.wls_lambda = wls_config['lambda']
            self.wls_sigma = wls_config['sigma']

        if self.use_wls_filter:
            try:
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
            except AttributeError:
                warnings.warn(
                    'OpenCV version does not support WLS filter. Disabling WLS filter. '
                    'Make sure you have opencv-contrib-python installed. '
                    'If not, run "pip uninstall opencv-python && pip install opencv-contrib-python -U"'
                )
                self.use_wls_filter = False

        self.msgs = dict()

        XoutFrames.__init__(self, frames=frames, fps=fps)
        Clickable.__init__(self, decay_step=int(self.fps))

    def on_callback(self, packet) -> None:
        if self.ir_settings['auto_mode']:
            self._auto_ir_search(packet.frame)

    def visualize(self, packet: DepthPacket):
        frame = packet.frame
        disparity_frame = (frame * self.multiplier).astype(np.uint8)
        try:
            mono_frame = packet.mono_frame.getCvFrame()
        except AttributeError:
            mono_frame = None

        stereo_config = self._visualizer.config.stereo

        if self.use_wls_filter or stereo_config.wls_filter:
            self.wls_filter.setLambda(self.wls_lambda or stereo_config.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma or stereo_config.wls_sigma)
            disparity_frame = self.wls_filter.filter(disparity_frame, mono_frame)

        colorize = self.colorize or stereo_config.colorize
        if self.colormap is not None:
            colormap = self.colormap
        else:
            colormap = stereo_config.colormap
            colormap[0] = [0, 0, 0]  # Invalidate pixels 0 to be black

        if mono_frame is not None and disparity_frame.ndim == 2 and mono_frame.ndim == 3:
            disparity_frame = disparity_frame[..., np.newaxis]

        if colorize == StereoColor.GRAY:
            packet.frame = disparity_frame
        elif colorize == StereoColor.RGB:
            packet.frame = cv2.applyColorMap(disparity_frame, colormap)
        elif colorize == StereoColor.RGBD:
            packet.frame = cv2.applyColorMap(
                (disparity_frame * 1.0 + mono_frame * 0.5).astype(np.uint8), colormap
            )

        if self._visualizer.config.output.clickable:
            cv2.namedWindow(self.name)
            cv2.setMouseCallback(self.name, self.on_click_callback, param=[disparity_frame])

            if self.buffer:
                x, y = self.buffer[2]
                text = f'{self.buffer[1]}'  # Disparity value
                if packet.depth_map is not None:
                    text = f"{packet.depth_map[y, x] / 1000 :.2f} m"

                self._visualizer.add_circle(coords=(x, y), radius=3, color=(255, 255, 255), thickness=-1)
                self._visualizer.add_text(text=text, coords=(x, y - 10))

        super().visualize(packet)

    def xstreams(self) -> List[StreamXout]:
        if self.mono_frames is None:
            return [self.frames]
        return [self.frames, self.mono_frames]

    def new_msg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams:
            return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()

        if name == self.frames.name:
            self.msgs[seq][name] = msg
        elif name == self.mono_frames.name:
            self.msgs[seq][name] = msg
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if len(self.msgs[seq]) == len(self.xstreams()):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            mono_frame = None
            if self.mono_frames is not None:
                mono_frame = self.msgs[seq][self.mono_frames.name]

            packet = DepthPacket(
                self.get_packet_name(),
                img_frame=self.msgs[seq][self.frames.name],
                mono_frame=mono_frame,
                visualizer=self._visualizer
            )
            self.queue.put(packet, block=False)

            new_msgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    new_msgs[name] = msg
            self.msgs = new_msgs

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

        if len(self._metrics_buffer['fill_rate']) < max(self.fps, 30):
            return False

        if candidate_idx >= len(candidate_pairs):
            # We have exhausted all candidates
            best_idx = np.argmax(self._y)
            self._converged_metric_value = self._y[best_idx]
            self._dot_projector_brightness, self._flood_brightness = self._X[best_idx]
            self._reset_buffers()
            self._auto_ir_converged = True
            self._checking_neighbourhood = False

            logging.debug(f'Auto IR converged: dot projector - {self._dot_projector_brightness}mA, '
                          f'flood - {self._flood_brightness}mA')
        else:
            self._dot_projector_brightness, self._flood_brightness = candidate_pairs[candidate_idx]

        self._update_ir()

        if self._auto_ir_converged:
            return False

        # Skip first half second of frames to allow for auto exposure to settle down
        fill_rate_avg = np.mean(self._metrics_buffer['fill_rate'][int(self.fps // 2):])

        self._X.append([self._dot_projector_brightness, self._flood_brightness])
        self._y.append(fill_rate_avg)

        self._metrics_buffer['fill_rate'].clear()
        return True

    def _check_consistency(self, frame):
        fill_rate = np.count_nonzero(frame) / frame.size
        self._metrics_buffer['fill_rate'].append(fill_rate)

        if len(self._metrics_buffer['fill_rate']) < max(self.fps, 30):
            return

        fill_rate_avg = np.mean(self._metrics_buffer['fill_rate'])
        self._metrics_buffer['fill_rate'].clear()

        if fill_rate_avg < self._converged_metric_value * 0.85:
            self._auto_ir_converged = False
            self._checking_neighbourhood = True
            self._neighbourhood_pairs = np.unique([
                [np.clip(self._dot_projector_brightness + i, 0, 1200), np.clip(self._flood_brightness + j, 0, 1500)]
                for i, j in itertools.product([-300, 300], [-375, 375])
            ], axis=0)
            self._neighbour_idx = 0

    def _update_ir(self):
        self.device.setIrLaserDotProjectorBrightness(self._dot_projector_brightness)
        self.device.setIrFloodLightBrightness(self._flood_brightness)

    def _reset_buffers(self):
        self._X, self._y = [], []
        del self._metrics_buffer['fill_rate']

import itertools
import logging
import warnings
from collections import defaultdict
from functools import cached_property
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

IR_FLOOD_LIMIT = 765
IR_FLOOD_STEP = IR_FLOOD_LIMIT / 4
IR_DOT_LIMIT = 1200
IR_DOT_STEP = IR_DOT_LIMIT / 4


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
                 ir_settings: dict = None,
                 confidence_map: StreamXout = None):
        self.mono_frames = mono_frames
        self.multiplier = disp_factor
        self.fps = fps
        self.name = 'Disparity'
        self.device = device

        self.confidence_map = confidence_map
        self.fig, self.axes = None, None  # for depth score visualization

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
        self._candidate_pairs = list(itertools.product(np.arange(0, IR_DOT_LIMIT + 1, IR_DOT_STEP),
                                                       np.arange(0, IR_FLOOD_LIMIT + 1, IR_FLOOD_STEP)))
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

        stereo_config = self._visualizer.config.stereo

        if self.use_wls_filter or stereo_config.wls_filter:
            self.wls_filter.setLambda(self.wls_lambda or stereo_config.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma or stereo_config.wls_sigma)
            disparity_frame = self.wls_filter.filter(disparity_frame, packet.mono_frame.getCvFrame())

        colorize = self.colorize or stereo_config.colorize
        if self.colormap is not None:
            colormap = self.colormap
        else:
            colormap = stereo_config.colormap
            colormap[0] = [0, 0, 0]  # Invalidate pixels 0 to be black

        if colorize == StereoColor.GRAY:
            packet.frame = disparity_frame
        elif colorize == StereoColor.RGB:
            packet.frame = cv2.applyColorMap(disparity_frame, colormap)
        elif colorize == StereoColor.RGBD:
            packet.frame = cv2.applyColorMap(
                (disparity_frame * 0.5 + packet.mono_frame.getCvFrame() * 0.5).astype(np.uint8), colormap
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

        if self._visualizer.config.stereo.depth_score and packet.confidence_map:
            self.fig.canvas.draw()
            self.axes.clear()
            self.axes.hist(255 - packet.confidence_map.getData(), bins=3, color='blue', alpha=0.5)
            self.axes.set_title(f'Depth score: {packet.depth_score:.2f}')
            self.axes.set_xlabel('Depth score')
            self.axes.set_ylabel('Frequency')

            # self.axes.text(0.5, 0.9, f'Overall depth score: {packet.depth_score:.2f}', ha='center', va='center',
            #                transform=self.axes.transAxes, fontsize=20)
            # Convert plot to numpy array
            img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(f'Depth score ({self.name})', img)

        super().visualize(packet)

    def xstreams(self) -> List[StreamXout]:
        streams = [self.frames]
        if self.mono_frames is not None:
            streams.append(self.mono_frames)
        if self.confidence_map is not None:
            streams.append(self.confidence_map)

        return streams

    def setup_visualize(self,
                        visualizer: 'Visualizer',
                        visualizer_enabled: bool,
                        name: str = None
                        ) -> None:
        super().setup_visualize(visualizer, visualizer_enabled, name)

        if self.confidence_map:
            from matplotlib import pyplot as plt
            self.fig, self.axes = plt.subplots(1, 1, figsize=(5, 2), constrained_layout=False)

    @cached_property
    def stream_names(self) -> List[str]:
        return [s.name for s in self.xstreams()]

    def new_msg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams:
            return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()

        if name in self.stream_names:
            self.msgs[seq][name] = msg
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if len(self.msgs[seq]) == len(self.xstreams()):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            mono_frame, confidence_map = None, None
            if self.mono_frames is not None:
                mono_frame = self.msgs[seq][self.mono_frames.name]
            if self.confidence_map is not None:
                confidence_map = self.msgs[seq][self.confidence_map.name]

            packet = DepthPacket(
                self.get_packet_name(),
                img_frame=self.msgs[seq][self.frames.name],
                confidence_map=confidence_map,
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

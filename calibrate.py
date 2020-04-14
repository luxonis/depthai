from contextlib import contextmanager

import depthai
from depthai_helpers.calibration_utils import *
from depthai_helpers import utils
import argparse
from argparse import ArgumentParser
import time
import numpy as np
import os
from pathlib import Path
import shutil
import consts.resource_paths
import json

use_cv = True
try:
    import cv2
except ImportError:
    use_cv = False


def parse_args():
    epilog_text = '''
    Captures and processes images for disparity depth calibration, generating a `dephtai.calib` file
    that should be loaded when initializing depthai. By default, captures one image across 13 polygon positions.

    Image capture requires the use of a printed 7x9 OpenCV checkerboard applied to a flat surface (ex: sturdy cardboard).
    When taking photos, ensure the checkerboard fits within both the left and right image displays. The board does not need
    to fit within each drawn red polygon shape, but it should mimic the display of the polygon.

    If the checkerboard does not fit within a captured image, the calibration will generate an error message with instructions
    on re-running calibration for polygons w/o valid checkerboard captures.

    The script requires a RMS error < 1.0 to generate a calibration file. If RMS exceeds this threshold, an error is displayed.

    Example usage:

    Run calibration with a checkerboard square size of 2.35 cm:
    python calibrate.py -s 2.35

    Run calibration for only the first and 3rd polygon positions:
    python calibrate.py -p 0 2

    Only run image processing (not image capture) w/ a 2.35 cm square size. Requires a set of polygon images:
    python calibrate.py -s 2.35 -m process

    Delete all existing images before starting image capture:
    python calibrate.py -i delete

    Capture 3 images per polygon:
    python calibrate.py -c 3

    Pass thru pipeline config options:
    python calibrate.py -co '{"board_config": {"swap_left_and_right_cameras": true, "left_to_right_distance_cm": 7.5}}'
    '''
    parser = ArgumentParser(epilog=epilog_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--count", default=1,
                        type=int, required=False,
                        help="Number of images per polygon to capture. Default is 1.")
    parser.add_argument("-s", "--square_size_cm", default="2.5",
                        type=float, required=False,
                        help="Square size of calibration pattern used in centimeters. Default is 2.5.")
    parser.add_argument("-i", "--image_op", default="modify",
                        type=str, required=False,
                        help="Whether existing images should be modified or all images should be deleted before running image capture. The default is 'modify'. Change to 'delete' to delete all image files.")
    parser.add_argument("-m", "--mode", default=['capture','process'], nargs='*',
                        type=str, required=False,
                        help="Space-separated list of calibration options to run. By default, executes the full 'capture process' pipeline. To execute a single step, enter just that step (ex: 'process').")
    parser.add_argument("-co", "--config_overwrite", default=None,
                        type=str, required=False,
                        help="JSON-formatted pipeline config object. This will be override defaults used in this script.")
    parser.add_argument("-fv", "--field-of-view", default=71.86, type=float,
                        help="Horizontal field of view (HFOV) for the stereo cameras in [deg]")
    parser.add_argument("-b", "--baseline", default=9.0, type=float,
                        help="Left/Right camera baseline in [cm]")
    parser.add_argument("-w", "--no-swap-lr", dest="swap_lr", default=True, action="store_false",
                        help="Do not swap the Left and Right cameras")

    options = parser.parse_args()

    return options


def find_chessboard(frame):
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    board_detected, _ = cv2.findChessboardCorners(small_frame, (9, 6), chessboard_flags)
    return board_detected


class Main:
    cmd_file = consts.resource_paths.device_cmd_fpath
    polygons = None
    current_polygon = 0
    images_captured_polygon = 0
    images_captured = 0

    def __init__(self):
        self.args = vars(parse_args())
        self.config = {
            'streams': ['left', 'right'],
            'depth':
                {
                    'calibration_file': consts.resource_paths.calib_fpath,
                    # 'type': 'median',
                    'padding_factor': 0.3
                },
            'ai':
                {
                    'blob_file': consts.resource_paths.blob_fpath,
                    'blob_file_config': consts.resource_paths.blob_config_fpath
                },
            'board_config':
                {
                    'swap_left_and_right_cameras': self.args['swap_lr'],
                    'left_fov_deg':  self.args['field_of_view'],
                    'left_to_right_distance_cm': self.args['baseline'],
                }
        }
        if self.args['config_overwrite']:
            utils.merge(json.loads(self.args['config_overwrite']), self.config)
            print("Merged Pipeline config with overwrite", self.config)
        self.total_images = self.args['count'] * len(setPolygonCoordinates(1000, 600))  # random polygons for count
        print("Using Arguments=", self.args)

    @contextmanager
    def get_pipeline(self):
        # Possible to try and reboot?
        # The following doesn't work (have to manually hit switch on device)
        # depthai.reboot_device
        # time.sleep(1)
        if not depthai.init_device(cmd_file=self.cmd_file):
            raise RuntimeError("Unable to initialize device. Try to reset it")

        pipeline = depthai.create_pipeline(self.config)

        if pipeline is None:
            raise RuntimeError("Unable to create pipeline")

        try:
            yield pipeline
        finally:
            del pipeline

    def parse_frame(self, frame, stream_name):
        if not find_chessboard(frame):
            return False

        filename = image_filename(stream_name, self.current_polygon, self.images_captured)
        cv2.imwrite("dataset/{}/{}".format(stream_name, filename), frame)
        print("py: Saved image as: " + str(filename))
        return True

    def capture_images(self):
        max_frames_per_capure = 4
        capture_counter = 0
        finished = False
        capturing = False
        captured_left = False
        captured_right = False
        with self.get_pipeline() as pipeline:
            while not finished:
                _, data_list = pipeline.get_available_nnet_and_data_packets()
                for packet in data_list:
                    frame = packet.getData()
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                    if self.polygons is None:
                        height, width, _ = frame.shape
                        self.polygons = setPolygonCoordinates(height, width)
                        print("Starting image capture. Press the [ESC] key to abort.")
                        print("Will take {} total images, {} per each polygon.".format(self.total_images,
                                                                                       self.args['count']))

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        print("py: Calibration has been interrupted!")
                        raise SystemExit(0)

                    if key == ord(" "):
                        capturing = True
                        capture_counter = 0

                    if capturing and packet.stream_name == 'left' and not captured_left:
                        captured_left = self.parse_frame(frame, packet.stream_name)
                    elif capturing and packet.stream_name == 'right' and not captured_right:
                        captured_right = self.parse_frame(frame, packet.stream_name)

                    if captured_left and captured_right:
                        self.images_captured += 1
                        self.images_captured_polygon += 1
                        captured_left = False
                        captured_right = False
                        capturing = False

                        if self.images_captured_polygon == self.args['count']:
                            self.images_captured_polygon = 0
                            self.current_polygon += 1

                            if self.current_polygon == len(self.polygons):
                                finished = True
                                break
                    elif capturing:
                        capture_counter += 1
                        if capture_counter > max_frames_per_capure:
                            print("py: Stopping the capture, unable to find chessboard! Fix position and press spacebar again")
                            capturing = False

                    has_success = (packet.stream_name == "left" and captured_left) or \
                                  (packet.stream_name == "right" and captured_right)
                    cv2.putText(
                        frame,
                        "Align cameras with callibration board and press spacebar to capture the image",
                        (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0)
                    )
                    cv2.putText(
                        frame,
                        "Polygon Position: {}. Captured {} of {} images.".format(
                            self.current_polygon, self.images_captured, self.total_images
                        ),
                        (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0)
                    )
                    if self.polygons is not None:
                        cv2.polylines(
                            frame, np.array([self.polygons[self.current_polygon]]),
                            True, (0, 255, 0) if has_success else (0, 0, 255), 4
                        )

                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # we don't need full resolution
                    cv2.imshow(packet.stream_name, small_frame)

    def calibrate(self):
        print("Starting image processing")
        cal_data = StereoCalibration()
        try:
            cal_data.calibrate("dataset", self.args['square_size_cm'], "./resources/depthai.calib")
        except AssertionError as e:
            print("[ERROR] " + str(e))
            raise SystemExit(1)

    def run(self):
        if 'capture' in self.args['mode']:
            try:
                if self.args['image_op'] == 'delete':
                    shutil.rmtree('dataset/')
                Path("dataset/left").mkdir(parents=True, exist_ok=True)
                Path("dataset/right").mkdir(parents=True, exist_ok=True)
            except OSError:
                print("An error occurred trying to create image dataset directories!")
                raise
            self.capture_images()
        if 'process' in self.args['mode']:
            self.calibrate()
        print('py: DONE.')


if __name__ == "__main__":
    Main().run()

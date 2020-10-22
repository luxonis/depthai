#!/usr/bin/env python3

import platform
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
from depthai_helpers.config_manager import BlobManager
from depthai_helpers.version_check import check_depthai_version
check_depthai_version()

use_cv = True
try:
    import cv2
except ImportError:
    use_cv = False

on_embedded = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')


def parse_args():
    epilog_text = '''
    Captures and processes images for disparity depth calibration, generating a `depthai.calib` file
    that should be loaded when initializing depthai. By default, captures one image for each of the 13 calibration target poses.

    Image capture requires the use of a printed 6x9 OpenCV checkerboard target applied to a flat surface (ex: sturdy cardboard).
    When taking photos, ensure the checkerboard fits within both the left and right image displays. The board does not need
    to fit within each drawn red polygon shape, but it should mimic the display of the polygon.

    If the calibration checkerboard corners cannot be found, the user will be prompted to try that calibration pose again.

    The script requires a RMS error < 1.0 to generate a calibration file. If RMS exceeds this threshold, an error is displayed.
    An average epipolar error of <1.5 is considered to be good, but not required. 

    Example usage:

    Run calibration with a checkerboard square size of 3.0 cm and baseline of 7.5cm:
    python3 calibrate.py -s 3.0 -b 7.5

    Only run image processing only with same board setup. Requires a set of saved capture images:
    python3 calibrate.py -s 3.0 -b 7.5 -m process
    
    Change Left/Right baseline to 15cm and swap Left/Right cameras:
    python3 calibrate.py -b 15 -w False

    Delete all existing images before starting image capture:
    python3 calibrate.py -i delete

    Pass thru pipeline config options:
    python3 calibrate.py -co '{"board_config": {"swap_left_and_right_cameras": true, "left_to_right_distance_cm": 7.5}}'
    '''
    parser = ArgumentParser(epilog=epilog_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--count", default=1,
                        type=int, required=False,
                        help="Number of images per polygon to capture. Default: 1.")
    parser.add_argument("-s", "--square_size_cm", default="2.5",
                        type=float, required=False,
                        help="Square size of calibration pattern used in centimeters. Default: 2.5cm.")
    parser.add_argument("-i", "--image_op", default="modify",
                        type=str, required=False,
                        help="Whether existing images should be modified or all images should be deleted before running image capture. The default is 'modify'. Change to 'delete' to delete all image files.")
    parser.add_argument("-m", "--mode", default=['capture','process'], nargs='*',
                        type=str, required=False,
                        help="Space-separated list of calibration options to run. By default, executes the full 'capture process' pipeline. To execute a single step, enter just that step (ex: 'process').")
    parser.add_argument("-co", "--config_overwrite", default=None,
                        type=str, required=False,
                        help="JSON-formatted pipeline config object. This will be override defaults used in this script.")
    parser.add_argument("-brd", "--board", default=None, type=str,
                        help="BW1097, BW1098OBC - Board type from resources/boards/ (not case-sensitive). "
                            "Or path to a custom .json board config. Mutually exclusive with [-fv -b -w]")
    parser.add_argument("-fv", "--field-of-view", default=None, type=float,
                        help="Horizontal field of view (HFOV) for the stereo cameras in [deg]. Default: 71.86deg.")
    parser.add_argument("-b", "--baseline", default=None, type=float,
                        help="Left/Right camera baseline in [cm]. Default: 9.0cm.")
    parser.add_argument("-w", "--no-swap-lr", dest="swap_lr", default=None, action="store_false",
                        help="Do not swap the Left and Right cameras.")
    parser.add_argument("-debug", "--dev_debug", default=None, action='store_true',
                        help="Used by board developers for debugging.")
    parser.add_argument("-iv", "--invert-vertical", dest="invert_v", default=False, action="store_true",
                        help="Invert vertical axis of the camera for the display")
    parser.add_argument("-ih", "--invert-horizontal", dest="invert_h", default=False, action="store_true",
                        help="Invert horizontal axis of the camera for the display")

    options = parser.parse_args()

    if (options.board is not None) and ((options.field_of_view is not None)
                                     or (options.baseline      is not None)
                                     or (options.swap_lr       is not None)):
        parser.error("[-brd] is mutually exclusive with [-fv -b -w]")

    # Set some defaults after the above check
    if options.field_of_view is None: options.field_of_view = 71.86
    if options.baseline      is None: options.baseline = 9.0
    if options.swap_lr       is None: options.swap_lr = True

    return options


def find_chessboard(frame):
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    return cv2.findChessboardCorners(small_frame, (9, 6), chessboard_flags)[0] and \
           cv2.findChessboardCorners(frame, (9, 6), chessboard_flags)[0]


def ts(packet):
    return packet.getMetadata().getTimestamp()


class Main:
    output_scale_factor = 0.5
    cmd_file = consts.resource_paths.device_cmd_fpath
    polygons = None
    width = None
    height = None
    current_polygon = 0
    images_captured_polygon = 0
    images_captured = 0

    def __init__(self):
        self.args = vars(parse_args())
        blob_man_args = {
                        'cnn_model':'mobilenet-ssd',
                        'cnn_model2':'',
                        'model_compilation_target':'auto'
                    }
        shaves = 7
        cmx_slices = 7
        NN_engines = 1
        blobMan = BlobManager(blob_man_args, True , shaves, cmx_slices,NN_engines)

        self.config = {
            'streams':
                ['left', 'right'] if not on_embedded else
                [{'name': 'left', "max_fps": 10.0}, {'name': 'right', "max_fps": 10.0}],
            'depth':
                {
                    'calibration_file': consts.resource_paths.calib_fpath,
                    'padding_factor': 0.3
                },
            'ai':
                {
                    'blob_file': blobMan.blob_file,
                    'blob_file_config': blobMan.blob_file_config,
                    'shaves' : shaves,
                    'cmx_slices' : cmx_slices,
                    'NN_engines' : NN_engines,
                },
            'board_config':
                {
                    'swap_left_and_right_cameras': self.args['swap_lr'],
                    'left_fov_deg':  self.args['field_of_view'],
                    'left_to_right_distance_cm': self.args['baseline'],
                    'override_eeprom': True,
                    'stereo_center_crop': True,
                },
            'camera':
                {
                    'mono':
                    {
                        # 1280x720, 1280x800, 640x400 (binning enabled)
                        'resolution_h': 800,
                        'fps': 30.0,
                    },
                },
        }
        if self.args['board']:
            board_path = Path(self.args['board'])
            if not board_path.exists():
                board_path = Path(consts.resource_paths.boards_dir_path) / Path(self.args['board'].upper()).with_suffix('.json')
                if not board_path.exists():
                    raise ValueError('Board config not found: {}'.format(board_path))
            with open(board_path) as fp:
                board_config = json.load(fp)
            utils.merge(board_config, self.config)
        if self.args['config_overwrite']:
            utils.merge(json.loads(self.args['config_overwrite']), self.config)
            print("Merged Pipeline config with overwrite", self.config)
        if self.args['dev_debug']:
            self.cmd_file = ''
            print('depthai will not load cmd file into device.')
        self.total_images = self.args['count'] * len(setPolygonCoordinates(1000, 600))  # random polygons for count
        print("Using Arguments=", self.args)

    @contextmanager
    def get_pipeline(self):
        # Possible to try and reboot?
        # The following doesn't work (have to manually hit switch on device)
        # depthai.reboot_device
        # time.sleep(1)

        pipeline = None

        try:
            device = depthai.Device("", False)
            pipeline = device.create_pipeline(self.config)
        except RuntimeError:
            raise RuntimeError("Unable to initialize device. Try to reset it")

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

    def show_info_frame(self):
        info_frame = np.zeros((600, 1000, 3), np.uint8)
        print("Starting image capture. Press the [ESC] key to abort.")
        print("Will take {} total images, {} per each polygon.".format(self.total_images, self.args['count']))

        def show(position, text):
            cv2.putText(info_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

        show((25, 100), "Information about image capture:")
        show((25, 160), "Press the [ESC] key to abort.")
        show((25, 220), "Press the [spacebar] key to capture the image.")
        show((25, 300), "Polygon on the image represents the desired chessboard")
        show((25, 340), "position, that will provide best calibration score.")
        show((25, 400), "Will take {} total images, {} per each polygon.".format(self.total_images, self.args['count']))
        show((25, 550), "To continue, press [spacebar]...")

        cv2.imshow("info", info_frame)
        while True:
            key = cv2.waitKey(1)
            if key == ord(" "):
                cv2.destroyAllWindows()
                return
            elif key == 27 or key == ord("q"):  # 27 - ESC
                cv2.destroyAllWindows()
                raise SystemExit(0)

    def show_failed_capture_frame(self):
        width, height = int(self.width * self.output_scale_factor), int(self.height * self.output_scale_factor)
        info_frame = np.zeros((height, width, 3), np.uint8)
        print("py: Capture failed, unable to find chessboard! Fix position and press spacebar again")

        def show(position, text):
            cv2.putText(info_frame, text, position, cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0))

        show((50, int(height / 2 - 40)), "Capture failed, unable to find chessboard!")
        show((60, int(height / 2 + 40)), "Fix position and press spacebar again")

        # cv2.imshow("left", info_frame)
        # cv2.imshow("right", info_frame)
        cv2.imshow("left + right",info_frame)
        cv2.waitKey(2000)

    def capture_images(self):
        finished = False
        capturing = False
        captured_left = False
        captured_right = False
        tried_left = False
        tried_right = False
        recent_left = None
        recent_right = None
        with self.get_pipeline() as pipeline:
            while not finished:
                _, data_list = pipeline.get_available_nnet_and_data_packets()
                for packet in data_list:
                    if packet.stream_name == "left" and (recent_left is None or ts(recent_left) < ts(packet)):
                        recent_left = packet
                    elif packet.stream_name == "right" and (recent_right is None or ts(recent_right) < ts(packet)):
                        recent_right = packet

                if recent_left is None or recent_right is None:
                    continue

                key = cv2.waitKey(1)
                if key == 27 or key == ord("q"):
                    print("py: Calibration has been interrupted!")
                    raise SystemExit(0)

                if key == ord(" "):
                    capturing = True
                
                frame_list = []
                for packet in (recent_left, recent_right):
                    frame = packet.getData()
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                    if self.polygons is None:
                        self.height, self.width, _ = frame.shape
                        self.polygons = setPolygonCoordinates(self.height, self.width)

                    if capturing and abs(ts(recent_left) - ts(recent_right)) < 0.001:
                        if packet.stream_name == 'left' and not tried_left:
                            captured_left = self.parse_frame(frame, packet.stream_name)
                            tried_left = True
                        elif packet.stream_name == 'right' and not tried_right:
                            captured_right = self.parse_frame(frame, packet.stream_name)
                            tried_right = True

                    has_success = (packet.stream_name == "left" and captured_left) or \
                                  (packet.stream_name == "right" and captured_right)

                    if self.args['invert_v'] and self.args['invert_h']:
                        frame = cv2.flip(frame, -1)
                    elif self.args['invert_v']:
                        frame = cv2.flip(frame, 0)
                    elif self.args['invert_h']:
                        frame = cv2.flip(frame, 1)

                    cv2.putText(
                        frame,
                        "Polygon Position: {}. Captured {} of {} images.".format(
                            self.current_polygon + 1, self.images_captured, self.total_images
                        ),
                        (0, 700), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0)
                    )
                    if self.polygons is not None:
                        cv2.polylines(
                            frame, np.array([self.polygons[self.current_polygon]]),
                            True, (0, 255, 0) if has_success else (0, 0, 255), 4
                        )

                    small_frame = cv2.resize(frame, (0, 0), fx=self.output_scale_factor, fy=self.output_scale_factor)
                    # cv2.imshow(packet.stream_name, small_frame)
                    frame_list.append(small_frame)

                    if captured_left and captured_right:
                        self.images_captured += 1
                        self.images_captured_polygon += 1
                        capturing = False
                        tried_left = False
                        tried_right = False
                        captured_left = False
                        captured_right = False

                    elif tried_left and tried_right:
                        self.show_failed_capture_frame()
                        capturing = False
                        tried_left = False
                        tried_right = False
                        captured_left = False
                        captured_right = False
                        break

                    if self.images_captured_polygon == self.args['count']:
                        self.images_captured_polygon = 0
                        self.current_polygon += 1

                        if self.current_polygon == len(self.polygons):
                            finished = True
                            cv2.destroyAllWindows()
                            break
                
                # combine_img = np.hstack((frame_list[0], frame_list[1]))
                combine_img = np.vstack((frame_list[0], frame_list[1]))

                cv2.imshow("left + right",combine_img)
                frame_list.clear()

    def calibrate(self):
        print("Starting image processing")
        flags = [self.config['board_config']['stereo_center_crop']]
        cal_data = StereoCalibration()
        try:
            cal_data.calibrate("dataset", self.args['square_size_cm'], "./resources/depthai.calib", flags)
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
            self.show_info_frame()
            self.capture_images()
        if 'process' in self.args['mode']:
            self.calibrate()
        print('py: DONE.')


if __name__ == "__main__":
    Main().run()

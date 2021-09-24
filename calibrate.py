#!/usr/bin/env python3
import argparse
import json
import shutil
import traceback
from argparse import ArgumentParser
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

import depthai_helpers.calibration_utils as calibUtils

font = cv2.FONT_HERSHEY_SIMPLEX
debug = False
red = (255, 0, 0)
green = (0, 255, 0)


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def parse_args():
    epilog_text = '''
    Captures and processes images for disparity depth calibration, generating a `<device id>.json` file or `depthai_calib.json`
    that should be loaded when initializing depthai. By default, captures one image for each of the 8 calibration target poses.

    Image capture requires the use of a printed OpenCV charuco calibration target applied to a flat surface(ex: sturdy cardboard).
    Default board size used in this script is 22x16. However you can send a customized one too.
    When taking photos, ensure enough amount of markers are visible and images are crisp. 
    The board does not need to fit within each drawn red polygon shape, but it should mimic the display of the polygon.

    If the calibration checkerboard corners cannot be found, the user will be prompted to try that calibration pose again.

    The script requires a RMS error < 1.0 to generate a calibration file. If RMS exceeds this threshold, an error is displayed.
    An average epipolar error of <1.5 is considered to be good, but not required. 

    Example usage:

    Run calibration with a checkerboard square size of 3.0cm and marker size of 2.5cm  on board config file DM2CAM:
    python3 calibrate.py -s 3.0 -ms 2.5 -brd DM2CAM

    Only run image processing only with same board setup. Requires a set of saved capture images:
    python3 calibrate.py -s 3.0 -ms 2.5 -brd DM2CAM -m process
    
    Delete all existing images before starting image capture:
    python3 calibrate.py -i delete
    '''
    parser = ArgumentParser(
        epilog=epilog_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--count", default=1, type=int, required=False,
                        help="Number of images per polygon to capture. Default: 1.")
    parser.add_argument("-s", "--squareSizeCm", type=float, required=True,
                        help="Square size of calibration pattern used in centimeters. Default: 2.0cm.")
    parser.add_argument("-ms", "--markerSizeCm", type=float, required=False,
                        help="Marker size in charuco boards.")
    parser.add_argument("-db", "--defaultBoard", default=False, action="store_true",
                        help="Calculates the -ms parameter automatically based on aspect ratio of charuco board in the repository")
    parser.add_argument("-nx", "--squaresX", default="11", type=int, required=False,
                        help="number of chessboard squares in X direction in charuco boards.")
    parser.add_argument("-ny", "--squaresY", default="8", type=int, required=False,
                        help="number of chessboard squares in Y direction in charuco boards.")
    parser.add_argument("-rd", "--rectifiedDisp", default=True, action="store_false",
                        help="Display rectified images with lines drawn for epipolar check")
    parser.add_argument("-drgb", "--disableRgb", default=False, action="store_true",
                        help="Disable rgb camera Calibration")
    parser.add_argument("-slr", "--swapLR", default=False, action="store_true",
                        help="Interchange Left and right camera port.")
    parser.add_argument("-m", "--mode", default=['capture', 'process'], nargs='*', type=str, required=False,
                        help="Space-separated list of calibration options to run. By default, executes the full 'capture process' pipeline. To execute a single step, enter just that step (ex: 'process').")
    parser.add_argument("-brd", "--board", default=None, type=str, required=True,
                        help="BW1097, BW1098OBC - Board type from resources/boards/ (not case-sensitive). "
                        "Or path to a custom .json board config. Mutually exclusive with [-fv -b -w]")
    parser.add_argument("-iv", "--invertVertical", dest="invert_v", default=False, action="store_true",
                        help="Invert vertical axis of the camera for the display")
    parser.add_argument("-ih", "--invertHorizontal", dest="invert_h", default=False, action="store_true",
                        help="Invert horizontal axis of the camera for the display")
    parser.add_argument("-ep", "--maxEpiploarError", default="1.0", type=float, required=False,
                        help="Sets the maximum epiploar allowed with rectification")
    parser.add_argument("-cm", "--cameraMode", default="perspective", type=str,
                        required=False, help="Choose between perspective and Fisheye")
    parser.add_argument("-rlp", "--rgbLensPosition", default=135, type=int,
                        required=False, help="Set the manual lens position of the camera for calibration")
    parser.add_argument("-fps", "--fps", default=30, type=int,
                        required=False, help="Set capture FPS for all cameras. Default: %(default)s")
    
    options = parser.parse_args()

    # Set some extra defaults, `-brd` would override them
    if options.markerSizeCm is None:
        if options.defaultBoard:
            options.markerSizeCm = options.squareSizeCm * 0.75
        else:
            raise argparse.ArgumentError(options.markerSizeCm, "-ms / --markerSizeCm needs to be provided (you can use -db / --defaultBoard if using calibration board from this repository or calib.io to calculate -ms automatically)")
    if options.squareSizeCm < 2.2:
        raise argparse.ArgumentTypeError("-s / --squareSizeCm needs to be greater than 2.2 cm")
        
    return options


class Main:
    output_scale_factor = 0.5
    polygons = None
    width = None
    height = None
    current_polygon = 0
    images_captured_polygon = 0
    images_captured = 0

    def __init__(self):
        self.args = parse_args()

        self.aruco_dictionary = cv2.aruco.Dictionary_get(
            cv2.aruco.DICT_4X4_1000)
        self.focus_value = self.args.rgbLensPosition
        if self.args.board:
            board_path = Path(self.args.board)
            if not board_path.exists():
                board_path = (Path(__file__).parent / 'resources/boards' / self.args.board.upper()).with_suffix('.json').resolve()
                if not board_path.exists():
                    raise ValueError(
                        'Board config not found: {}'.format(board_path))
            with open(board_path) as fp:
                self.board_config = json.load(fp)
        # TODO: set the total images
        # random polygons for count
        self.total_images = self.args.count * \
            len(calibUtils.setPolygonCoordinates(1000, 600))
        if debug:
            print("Using Arguments=", self.args)

        pipeline = self.create_pipeline()
        self.device = dai.Device(pipeline)

        self.left_camera_queue = self.device.getOutputQueue("left", 30, True)
        self.right_camera_queue = self.device.getOutputQueue("right", 30, True)
        if not self.args.disableRgb:
            self.rgb_camera_queue = self.device.getOutputQueue("rgb", 30, True)

    def is_markers_found(self, frame):
        marker_corners, _, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dictionary)
        print("Markers count ... {}".format(len(marker_corners)))
        return not (len(marker_corners) < self.args.squaresX*self.args.squaresY / 4)

    def test_camera_orientation(self, frame_l, frame_r):
        marker_corners_l, id_l, _ = cv2.aruco.detectMarkers(
            frame_l, self.aruco_dictionary)
        marker_corners_r, id_r, _ = cv2.aruco.detectMarkers(
            frame_r, self.aruco_dictionary)

        for i, left_id in enumerate(id_l):
            idx = np.where(id_r == left_id)
            print(idx)
            if idx[0].size == 0:
                continue
            for left_corner, right_corner in zip(marker_corners_l[i], marker_corners_r[idx[0][0]]):
                if left_corner[0][0] - right_corner[0][0] < 0:
                    return False
        return True

    def create_pipeline(self):
        pipeline = dai.Pipeline()

        cam_left = pipeline.createMonoCamera()
        cam_right = pipeline.createMonoCamera()

        xout_left = pipeline.createXLinkOut()
        xout_right = pipeline.createXLinkOut()

        if self.args.swapLR:
            cam_left.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            cam_right.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                
        cam_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_800_P)
        cam_left.setFps(self.args.fps)

        cam_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_800_P)
        cam_right.setFps(self.args.fps)

        xout_left.setStreamName("left")
        cam_left.out.link(xout_left.input)

        xout_right.setStreamName("right")
        cam_right.out.link(xout_right.input)

        if not self.args.disableRgb:
            rgb_cam = pipeline.createColorCamera()
            rgb_cam.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_4_K)
            rgb_cam.setInterleaved(False)
            rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            rgb_cam.setIspScale(1, 3)
            rgb_cam.initialControl.setManualFocus(self.focus_value)
            rgb_cam.setFps(self.args.fps)

            xout_rgb_isp = pipeline.createXLinkOut()
            xout_rgb_isp.setStreamName("rgb")
            rgb_cam.isp.link(xout_rgb_isp.input)        

        return pipeline

    def parse_frame(self, frame, stream_name):
        if not self.is_markers_found(frame):
            return False

        filename = calibUtils.image_filename(
            stream_name, self.current_polygon, self.images_captured)
        cv2.imwrite("dataset/{}/{}".format(stream_name, filename), frame)
        print("py: Saved image as: " + str(filename))
        return True

    def show_info_frame(self):
        info_frame = np.zeros((600, 1000, 3), np.uint8)
        print("Starting image capture. Press the [ESC] key to abort.")
        print("Will take {} total images, {} per each polygon.".format(
            self.total_images, self.args.count))

        def show(position, text):
            cv2.putText(info_frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

        show((25, 100), "Information about image capture:")
        show((25, 160), "Press the [ESC] key to abort.")
        show((25, 220), "Press the [spacebar] key to capture the image.")
        show((25, 300), "Polygon on the image represents the desired chessboard")
        show((25, 340), "position, that will provide best calibration score.")
        show((25, 400), "Will take {} total images, {} per each polygon.".format(
            self.total_images, self.args.count))
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
        width, height = int(
            self.width * self.output_scale_factor), int(self.height * self.output_scale_factor)
        info_frame = np.zeros((height, width, 3), np.uint8)
        print("py: Capture failed, unable to find chessboard! Fix position and press spacebar again")

        def show(position, text):
            cv2.putText(info_frame, text, position,
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0))

        show((50, int(height / 2 - 40)),
             "Capture failed, unable to find chessboard!")
        show((60, int(height / 2 + 40)), "Fix position and press spacebar again")

        # cv2.imshow("left", info_frame)
        # cv2.imshow("right", info_frame)
        cv2.imshow("left + rgb + right", info_frame)
        cv2.waitKey(2000)

    def show_failed_orientation(self):
        width, height = int(
            self.width * self.output_scale_factor), int(self.height * self.output_scale_factor)
        info_frame = np.zeros((height, width, 3), np.uint8)
        print("py: Capture failed, Swap the camera's ")

        def show(position, text):
            cv2.putText(info_frame, text, position,
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0))

        show((60, int(height / 2 - 40)), "Calibration failed, ")
        show((60, int(height / 2)), "Device might be held upside down!")
        show((60, int(height / 2)), "Or ports connected might be inverted!")
        show((60, int(height / 2 + 40)), "Fix orientation")
        show((60, int(height / 2 + 80)), "and start again")

        # cv2.imshow("left", info_frame)
        # cv2.imshow("right", info_frame)
        cv2.imshow("left + right", info_frame)
        cv2.waitKey(0)
        raise Exception(
            "Calibration failed, Camera Might be held upside down. start again!!")

    def capture_images(self):
        finished = False
        capturing = False
        captured_left = False
        captured_right = False
        captured_color = False
        tried_left = False
        tried_right = False
        tried_color = False
        recent_left = None
        recent_right = None
        recent_color = None
        # with self.get_pipeline() as pipeline:
        while not finished:
            current_left  = self.left_camera_queue.tryGet()
            current_right = self.right_camera_queue.tryGet()
            if not self.args.disableRgb:
                current_color = self.rgb_camera_queue.tryGet()
            else:
                current_color = None
            # recent_left = left_frame.getCvFrame()
            # recent_color = cv2.cvtColor(rgb_frame.getCvFrame(), cv2.COLOR_BGR2GRAY)
            if not current_left is None:
                recent_left = current_left
            if not current_right is None:
                recent_right = current_right
            if not current_color is None:
                recent_color = current_color

            if recent_left is None or recent_right is None or (recent_color is None and not self.args.disableRgb):
                print("Continuing...")
                continue

            recent_frames = [('left', recent_left), ('right', recent_right)]
            if not self.args.disableRgb:
                recent_frames.append(('rgb', recent_color))

            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                print("py: Calibration has been interrupted!")
                raise SystemExit(0)
            elif key == ord(" "):
                if debug:
                    print("setting capture true------------------------")
                capturing = True

            frame_list = []
            # left_frame = recent_left.getCvFrame()
            # rgb_frame = recent_color.getCvFrame()

            for packet in recent_frames:
                frame = packet[1].getCvFrame()
                # print(packet[0])
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                if packet[0] == 'rgb':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # print(frame.shape)
                if self.polygons is None:
                    self.height, self.width = frame.shape
                    print(self.height, self.width)
                    self.polygons = calibUtils.setPolygonCoordinates(
                        self.height, self.width)

                if debug:
                    print("Timestamp difference ---> l & rgb")
                lrgb_time = 0
                if not self.args.disableRgb:
                    lrgb_time = min([abs((recent_left.getTimestamp() - recent_color.getTimestamp()).microseconds), abs((recent_color.getTimestamp() - recent_left.getTimestamp()).microseconds)])
                lr_time = min([abs((recent_left.getTimestamp() - recent_right.getTimestamp()).microseconds), abs((recent_right.getTimestamp() - recent_left.getTimestamp()).microseconds)])

                if debug:
                    print(lrgb_time)
                    print(lr_time)

                if capturing and lrgb_time < 30000 and lr_time < 30000:
                    print("Capturing  ------------------------")
                    if packet[0] == 'left' and not tried_left:
                        captured_left = self.parse_frame(frame, packet[0])
                        tried_left = True
                        captured_left_frame = frame.copy()
                    elif packet[0] == 'rgb' and not tried_color and not self.args.disableRgb:
                        captured_color = self.parse_frame(frame, packet[0])
                        tried_color = True
                        captured_color_frame = frame.copy()
                    elif packet[0] == 'right' and not tried_right:
                        captured_right = self.parse_frame(frame, packet[0])
                        tried_right = True
                        captured_right_frame = frame.copy()


                has_success = (packet[0] == "left" and captured_left) or (packet[0] == "right" and captured_right)  or \
                    (packet[0] == "rgb" and captured_color)

                if self.args.invert_v and self.args.invert_h:
                    frame = cv2.flip(frame, -1)
                elif self.args.invert_v:
                    frame = cv2.flip(frame, 0)
                elif self.args.invert_h:
                    frame = cv2.flip(frame, 1)

                cv2.putText(
                    frame,
                    "Polygon Position: {}. Captured {} of {} {} images".format(
                        self.current_polygon + 1, self.images_captured, self.total_images, packet[0]
                    ),
                    (0, 700), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0)
                )
                if self.polygons is not None:
                    cv2.polylines(
                        frame, np.array([self.polygons[self.current_polygon]]),
                        True, (0, 255, 0) if captured_left else (0, 0, 255), 4
                    )

                small_frame = cv2.resize(
                    frame, (0, 0), fx=self.output_scale_factor, fy=self.output_scale_factor)
                # cv2.imshow(packet.stream_name, small_frame)
                frame_list.append(small_frame)

                if self.args.disableRgb:
                    captured_color = True
                if captured_left and captured_right and captured_color:
                    print(f"Images captured --> {self.images_captured}")
                    if not self.images_captured:
                        if not self.test_camera_orientation(captured_left_frame, captured_right_frame):
                            self.show_failed_orientation()
                        # if not self.test_camera_orientation(captured_left_frame, captured_color_frame):
                        #     self.show_failed_orientation()

                    self.images_captured += 1
                    self.images_captured_polygon += 1
                    capturing = False
                    tried_left = False
                    tried_right = False
                    tried_color = False
                    captured_left = False
                    captured_right = False
                    captured_color = False
                elif tried_left and tried_color:
                    self.show_failed_capture_frame()
                    capturing = False
                    tried_left = False
                    tried_right = False
                    tried_color = False
                    captured_left = False
                    captured_right = False
                    captured_color = False
                    break

                if self.images_captured_polygon == self.args.count:
                    self.images_captured_polygon = 0
                    self.current_polygon += 1

                    if self.current_polygon == len(self.polygons):
                        finished = True
                        cv2.destroyAllWindows()
                        break
            
            combine_img = None
            if not self.args.disableRgb:
                frame_list[2] = np.pad(frame_list[2], ((40, 0), (0,0)), 'constant', constant_values=0)
                combine_img = np.hstack((frame_list[0], frame_list[1], frame_list[2]))
            else:
                combine_img = np.vstack((frame_list[0], frame_list[1]))
            cv2.imshow("left + rgb + right", combine_img)
            frame_list.clear()

    def calibrate(self):
        print("Starting image processing")
        cal_data = calibUtils.StereoCalibration()
        dest_path = str(Path('resources').absolute())
        self.args.cameraMode = 'perspective' # hardcoded for now
        try:
            epiploar_error, epiploar_error_rRgb, calibData = cal_data.calibrate(self.dataset_path, self.args.squareSizeCm,
                 self.args.markerSizeCm, self.args.squaresX, self.args.squaresY, self.args.cameraMode, not self.args.disableRgb, self.args.rectifiedDisp)
            if epiploar_error > self.args.maxEpiploarError:
                image = create_blank(900, 512, rgb_color=red)
                text = "High L-r epiploar_error: " + str(epiploar_error)
                cv2.putText(image, text, (10, 250), font, 2, (0, 0, 0), 2)
                text = "Requires Recalibration "
                cv2.putText(image, text, (10, 300), font, 2, (0, 0, 0), 2)

                cv2.imshow("Result Image", image)
                cv2.waitKey(0)
                print("Requires Recalibration.....!!")
                raise SystemExit(1)
            elif epiploar_error_rRgb is not None and epiploar_error_rRgb > self.args.maxEpiploarError:
                image = create_blank(900, 512, rgb_color=red)
                text = "High RGB-R epiploar_error: " + str(epiploar_error_rRgb)
                cv2.putText(image, text, (10, 250), font, 2, (0, 0, 0), 2)
                text = "Requires Recalibration "
                cv2.putText(image, text, (10, 300), font, 2, (0, 0, 0), 2)

                cv2.imshow("Result Image", image)
                cv2.waitKey(0)
                print("Requires Recalibration.....!!")
                raise SystemExit(1)

            left = dai.CameraBoardSocket.LEFT
            right = dai.CameraBoardSocket.RIGHT
            if self.args.swapLR:
                left = dai.CameraBoardSocket.RIGHT
                right = dai.CameraBoardSocket.LEFT

            calibration_handler = dai.CalibrationHandler()
            calibration_handler.setBoardInfo(self.board_config['board_config']['name'], self.board_config['board_config']['revision'])

            calibration_handler.setCameraIntrinsics(left, calibData[2], 1280, 800)
            calibration_handler.setCameraIntrinsics(right, calibData[3], 1280, 800)
            measuredTranslation = [
                -self.board_config['board_config']['left_to_right_distance_cm'], 0.0, 0.0]
            calibration_handler.setCameraExtrinsics(
                left, right, calibData[5], calibData[6], measuredTranslation)

            calibration_handler.setDistortionCoefficients(left, calibData[9] )
            calibration_handler.setDistortionCoefficients(right, calibData[10])

            calibration_handler.setFov(left, self.board_config['board_config']['left_fov_deg'])
            calibration_handler.setFov(right, self.board_config['board_config']['left_fov_deg'])

            calibration_handler.setStereoLeft(
                left, calibData[0])
            calibration_handler.setStereoRight(
                right, calibData[1])

            if not self.args.disableRgb:
                calibration_handler.setCameraIntrinsics(dai.CameraBoardSocket.RGB, calibData[4], 1920, 1080)
                calibration_handler.setDistortionCoefficients(dai.CameraBoardSocket.RGB, calibData[11])
                calibration_handler.setFov(dai.CameraBoardSocket.RGB, self.board_config['board_config']['rgb_fov_deg'])
                calibration_handler.setLensPosition(dai.CameraBoardSocket.RGB, self.focus_value)

                measuredTranslation = [
                    self.board_config['board_config']['left_to_rgb_distance_cm'], 0.0, 0.0]
                calibration_handler.setCameraExtrinsics(
                    right, dai.CameraBoardSocket.RGB, calibData[7], calibData[8], measuredTranslation)
            
            resImage = None
            if not self.device.isClosed():
                dev_info = self.device.getDeviceInfo()
                mx_serial_id = dev_info.getMxId()
                calib_dest_path = dest_path + '/' + mx_serial_id + '.json'
                calibration_handler.eepromToJsonFile(calib_dest_path)
                is_write_succesful = False
                
                try:
                    is_write_succesful = self.device.flashCalibration(
                        calibration_handler)
                except:
                    print("Writing in except...")
                    is_write_succesful = self.device.flashCalibration(
                        calibration_handler)
                if is_write_succesful:
                    resImage = create_blank(900, 512, rgb_color=green)
                    text = "Calibration Succesful with"
                    cv2.putText(resImage, text, (10, 250),
                                font, 2, (0, 0, 0), 2)
                    text = "Epipolar error of " + str(epiploar_error)
                    cv2.putText(resImage, text, (10, 300),
                                font, 2, (0, 0, 0), 2)
                else:
                    resImage = create_blank(900, 512, rgb_color=red)
                    text = "EEprom Write Failed!! " + str(epiploar_error)
                    cv2.putText(resImage, text, (10, 250),
                                font, 2, (0, 0, 0), 2)
                    text = "Try recalibrating !!"
                    cv2.putText(resImage, text, (10, 300),
                                font, 2, (0, 0, 0), 2)
            else:
                calib_dest_path = dest_path + '/depthai_calib.json'
                # calibration_handler.eepromToJsonFile(calib_dest_path)
                resImage = create_blank(900, 512, rgb_color=red)
                text = "Calibratin succesful. " + str(epiploar_error)
                cv2.putText(resImage, text, (10, 250), font, 2, (0, 0, 0), 2)
                # text = "Device not found to write to EEPROM"
                # cv2.putText(resImage, text, (10, 300), font, 2, (0, 0, 0), 2)

            if resImage is not None:
                cv2.imshow("Result Image", resImage)
                cv2.waitKey(0)
        except AssertionError as e:
            print("[ERROR] " + str(e))
            raise SystemExit(1)

    def run(self):
        if 'capture' in self.args.mode:
            try:
                if Path('dataset').exists():
                    shutil.rmtree('dataset/')
                Path("dataset/left").mkdir(parents=True, exist_ok=True)
                Path("dataset/right").mkdir(parents=True, exist_ok=True)
                if not self.args.disableRgb:
                    Path("dataset/rgb").mkdir(parents=True, exist_ok=True)
            except OSError:
                traceback.print_exc()
                print("An error occurred trying to create image dataset directories!")
                raise SystemExit(1)
            self.show_info_frame()
            self.capture_images()
        self.dataset_path = str(Path("dataset").absolute())
        if 'process' in self.args.mode:
            self.calibrate()
        print('py: DONE.')


if __name__ == "__main__":
    Main().run()

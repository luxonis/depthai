#!/usr/bin/env python3
import argparse
import json
from pydoc import render_doc
import shutil
import traceback
from argparse import ArgumentParser
from pathlib import Path
import time
from datetime import datetime, timedelta
from collections import deque 
from scipy.spatial.transform import Rotation
import traceback
import itertools
import math

import cv2
from cv2 import resize
import depthai as dai
import numpy as np
import copy

import depthai_calibration.calibration_utils as calibUtils

font = cv2.FONT_HERSHEY_SIMPLEX
red = (255, 0, 0)
green = (0, 255, 0)

stringToCam = {
                'RGB'   : dai.CameraBoardSocket.CAM_A,
                'LEFT'  : dai.CameraBoardSocket.CAM_B,
                'RIGHT' : dai.CameraBoardSocket.CAM_C,
                'CAM_A' : dai.CameraBoardSocket.CAM_A,
                'CAM_B' : dai.CameraBoardSocket.CAM_B,
                'CAM_C' : dai.CameraBoardSocket.CAM_C,
                'CAM_D' : dai.CameraBoardSocket.CAM_D,
                'CAM_E' : dai.CameraBoardSocket.CAM_E,
                'CAM_F' : dai.CameraBoardSocket.CAM_F,
                'CAM_G' : dai.CameraBoardSocket.CAM_G,
                'CAM_H' : dai.CameraBoardSocket.CAM_H
                }


camToMonoRes = {
                'OV7251' : dai.MonoCameraProperties.SensorResolution.THE_480_P,
                'OV9282' : dai.MonoCameraProperties.SensorResolution.THE_800_P,
                }

camToRgbRes = {
                'IMX378' : dai.ColorCameraProperties.SensorResolution.THE_4_K,
                'IMX214' : dai.ColorCameraProperties.SensorResolution.THE_4_K,
                'OV9782' : dai.ColorCameraProperties.SensorResolution.THE_800_P,
                'IMX582' : dai.ColorCameraProperties.SensorResolution.THE_12_MP,
                'AR0234' : dai.ColorCameraProperties.SensorResolution.THE_1200_P,
                }

antibandingOpts = {
    'off': dai.CameraControl.AntiBandingMode.OFF,
    '50':  dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
    '60':  dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
}

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

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
    parser.add_argument("-c", "--count", default=3, type=int, required=False,
                        help="Number of images per polygon to capture. Default: 1.")
    parser.add_argument("-s", "--squareSizeCm", type=float, required=True,
                        help="Square size of calibration pattern used in centimeters. Default: 2.0cm.")
    parser.add_argument("-ms", "--markerSizeCm", type=float, required=False,
                        help="Marker size in charuco boards.")
    parser.add_argument("-db", "--defaultBoard", default=None, type=str,
                        help="Calculates the size of markers, numbers of squareX and squareY base on the choosing board from charuco_boards directory.")
    parser.add_argument("-nx", "--squaresX", default="11", type=int, required=False,
                        help="number of chessboard squares in X direction in charuco boards.")
    parser.add_argument("-ny", "--squaresY", default="8", type=int, required=False,
                        help="number of chessboard squares in Y direction in charuco boards.")
    parser.add_argument("-rd", "--rectifiedDisp", default=True, action="store_false",
                        help="Display rectified images with lines drawn for epipolar check")
    parser.add_argument("-m", "--mode", default=['capture', 'process'], nargs='*', type=str, required=False,
                        help="Space-separated list of calibration options to run. By default, executes the full 'capture process' pipeline. To execute a single step, enter just that step (ex: 'process').")
    parser.add_argument("-brd", "--board", default=None, type=str,
                        help="BW1097, BW1098OBC - Board type from resources/depthai_boards/boards (not case-sensitive). "
                        "Or path to a custom .json board config. Mutually exclusive with [-fv -b -w]")
    parser.add_argument("-iv", "--invertVertical", dest="invert_v", default=False, action="store_true",
                        help="Invert vertical axis of the camera for the display")
    parser.add_argument("-ih", "--invertHorizontal", dest="invert_h", default=False, action="store_true",
                        help="Invert horizontal axis of the camera for the display")
    parser.add_argument("-ep", "--maxEpiploarError", default="0.8", type=float, required=False,
                         help="Sets the maximum epiploar allowed with rectification. Default: %(default)s")
    parser.add_argument("-cm", "--cameraMode", default="perspective", type=str,
                        required=False, help="Choose between perspective and Fisheye")
    parser.add_argument('-rlp', '--rgbLensPosition', nargs='*', action=ParseKwargs, required=False, default={} , help="Set the manual lens position of the camera for calibration. Example -rlp rgb=135 night=135")
    parser.add_argument('-dsb', '--disableCamera', nargs='+', required=False, default=[] , help="Set which camera should be disabled. Example -dsb rgb left right")
    parser.add_argument("-cd", "--captureDelay", default=2, type=int,
                        required=False, help="Choose how much delay to add between pressing the key and capturing the image. Default: %(default)s")
    parser.add_argument("-fac", "--factoryCalibration", default=False, action="store_true",
                        help="Enable writing to Factory Calibration.")
    parser.add_argument("-osf", "--outputScaleFactor", type=float, default=0.5,
                        help="set the scaling factor for output visualization. Default: 0.5.")
    parser.add_argument('-fps', '--framerate', type=float, default=10,
                        help="FPS to set for all cameras. Default: %(default)s")
    parser.add_argument('-ab', '--antibanding', default='50', choices={'off', '50', '60'},
                        help="Set antibanding/antiflicker algo for lights that flicker at mains frequency. Default: %(default)s [Hz]")
    parser.add_argument('-scp', '--saveCalibPath', type=str, default="",
                        help="Save calibration file to this path")
    parser.add_argument('-dst', '--datasetPath', type=str, default="dataset",
                        help="Path to dataset used for processing images")
    parser.add_argument('-mdmp', '--minDetectedMarkersPercent', type=float, default=0.4,
                        help="Minimum percentage of detected markers to consider a frame valid")
    parser.add_argument('-mt', '--mouseTrigger', default=False, action="store_true",
                        help="Enable mouse trigger for image capture")
    parser.add_argument('-nic', '--noInitCalibration', default=False, action="store_true",
                        help="Don't take the board calibration for initialization but start with an empty one")
    parser.add_argument('-trc', '--traceLevel', type=int, default=0,
                        help="Set to trace the steps in calibration. Number from 1 to 5. If you want to display all, set trace number to 10.")
    parser.add_argument('-mst', '--minSyncTimestamp',  type=float, default=0.2,
                        help="Minimum time difference between pictures taken from different cameras.  Default: %(default)s ")
    parser.add_argument('-it', '--numPictures',  type=float, default=None,
                        help="Number of pictures taken.")
    parser.add_argument('-ebp', '--enablePolygonsDisplay', default=False, action="store_true",
                        help="Enable the display of polynoms.")
    parser.add_argument('-dbg', '--debugProcessingMode', default=False, action="store_true",
                        help="Enable processing of images without using the camera.")
    options = parser.parse_args()
    # Set some extra defaults, `-brd` would override them
    if options.defaultBoard is not None:
        try:
            board_name = options.defaultBoard
            try:
                board_name, _ = board_name.split(".")
            except:
                board_name = board_name
            _, size, charcuo_num = board_name.split("_")
            numX, numY = charcuo_num.split("x")
            options.squaresX = int(numX)
            options.squaresY = int(numY)
        except:
            raise argparse.ArgumentTypeError(options.defaultBoard, "Board name has not been found.")
    if options.markerSizeCm is None:
        options.markerSizeCm = options.squareSizeCm * 0.75
    if options.squareSizeCm < 2.2:
        raise argparse.ArgumentTypeError("-s / --squareSizeCm needs to be greater than 2.2 cm")
    if options.traceLevel == 1:
        print(f"Charuco board selected is: board_name = {board_name}, numX = {numX}, numY = {numY}, squareSize {options.squareSizeCm} cm, markerSize {options.markerSizeCm} cm")
    if options.debugProcessingMode:
        options.mode = "process"
        if options.board is None:
            raise argparse.ArgumentError(options.board, "Board name (-brd) of camera must be specified in case of using debug mode (-dbg).")
    return options

class HostSync:
    def __init__(self, deltaMilliSec):
        self.arrays = {}
        self.arraySize = 15
        self.recentFrameTs = None
        self.deltaMilliSec = timedelta(milliseconds=deltaMilliSec)
        # self.synced = queue.Queue()

    def remove(self, t1):
            return timedelta(milliseconds=500) < (self.recentFrameTs - t1)

    def add_msg(self, name, data, ts):
        if name not in self.arrays:
            self.arrays[name] = deque(maxlen=self.arraySize)
        # Add msg to array
        self.arrays[name].appendleft({'data': data, 'timestamp': ts})
        if self.recentFrameTs == None or self.recentFrameTs - ts:
            self.recentFrameTs = ts
    
    def clearQueues(self):
        print('Clearing Queues...')
        for name, msgList in self.arrays.items():
            self.arrays[name].clear()
            print(len(self.arrays[name]))

    def get_synced(self):
        synced = {}
        for name, msgList in self.arrays.items():
            if len(msgList) != self.arraySize:
                return False 

        for name, pivotMsgList in self.arrays.items():
            print('len(pivotMsgList)')
            print(len(pivotMsgList))
            pivotMsgListDuplicate = pivotMsgList
            while pivotMsgListDuplicate:
                currPivot = pivotMsgListDuplicate.popleft()
                synced[name] = currPivot['data']
                
                for subName, msgList in self.arrays.items():
                    print(f'len of {subName}')
                    print(len(msgList))
                    if name == subName:
                        continue
                    msgListDuplicate = msgList.copy()
                    while msgListDuplicate:
                        print(f'---len of dup {subName} is {len(msgListDuplicate)}')
                        currMsg = msgListDuplicate.popleft()
                        time_diff = abs(currMsg['timestamp'] - currPivot['timestamp'])
                        print(f'---Time diff is {time_diff} and delta is {self.deltaMilliSec}')
                        if time_diff < self.deltaMilliSec:
                            print(f'--------Adding {subName} to sync. Messages left is {len(msgListDuplicate)}')
                            synced[subName] = currMsg['data']
                            break
                    print(f'Size of Synced is {len(synced)} amd array size is {len(self.arrays)}')
                    if len(synced) == len(self.arrays):
                        self.clearQueues()
                        return synced

            # raise SystemExit(1)
            self.clearQueues()
            return False


class MessageSync:
    def __init__(self, num_queues, min_diff_timestamp, max_num_messages=4, min_queue_depth=3):
        self.num_queues = num_queues
        self.min_diff_timestamp = min_diff_timestamp
        self.max_num_messages = max_num_messages
        # self.queues = [deque() for _ in range(num_queues)]
        self.queues = dict()
        self.queue_depth = min_queue_depth
        # self.earliest_ts = {}

    def add_msg(self, name, msg):
        if name not in self.queues:
            self.queues[name] = deque(maxlen=self.max_num_messages)
        self.queues[name].append(msg)
        # if msg.getTimestampDevice() < self.earliest_ts:
        #     self.earliest_ts = {name: msg.getTimestampDevice()}

        # print('Queues: ', end='')
        # for name in self.queues.keys():
        #     print('\t: ', name, end='')
        #     print(self.queues[name], end=', ')
        #     print()
        # print()

    def get_synced(self):

        # Atleast 3 messages should be buffered
        min_len = min([len(queue) for queue in self.queues.values()])
        if min_len == 0:
            print('Status:', 'exited due to min len == 0', self.queues)
            return None

        # initializing list of list 
        queue_lengths = []
        for name in self.queues.keys():
            queue_lengths.append(range(0, len(self.queues[name])))
        permutations = list(itertools.product(*queue_lengths))
        # print ("All possible permutations are : " +  str(permutations))

        # Return a best combination after being atleast 3 messages deep for all queues
        min_ts_diff = None
        for indicies in permutations:
            tmp = {}
            i = 0
            for n in self.queues.keys():
                tmp[n] = indicies[i]
                i = i + 1
            indicies = tmp

            acc_diff = 0.0
            min_ts = None
            for name in indicies.keys():
                msg = self.queues[name][indicies[name]]
                if min_ts is None:
                    min_ts = msg.getTimestampDevice().total_seconds()
            for name in indicies.keys():
                msg = self.queues[name][indicies[name]]
                acc_diff = acc_diff + abs(min_ts - msg.getTimestampDevice().total_seconds())

            # Mark minimum
            if min_ts_diff is None or (acc_diff < min_ts_diff['ts'] and abs(acc_diff - min_ts_diff['ts']) > 0.03):
                min_ts_diff = {'ts': acc_diff, 'indicies': indicies.copy()}
                print('new minimum:', min_ts_diff, 'min required:', self.min_diff_timestamp)

            if min_ts_diff['ts'] < self.min_diff_timestamp:
                # Check if atleast 5 messages deep
                min_queue_depth = None
                for name in indicies.keys():
                    if min_queue_depth is None or indicies[name] < min_queue_depth:
                        min_queue_depth = indicies[name]
                if min_queue_depth >= self.queue_depth:
                    # Retrieve and pop the others
                    synced = {}
                    for name in indicies.keys():
                        synced[name] = self.queues[name][min_ts_diff['indicies'][name]]
                        # pop out the older messages
                        for i in range(0, min_ts_diff['indicies'][name]+1):
                            self.queues[name].popleft()
                    if self.traceLevel == 1:
                        print('Returning synced messages with error:', min_ts_diff['ts'], min_ts_diff['indicies'])
                    return synced


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
        self.traceLevel= self.args.traceLevel
        self.output_scale_factor = self.args.outputScaleFactor
        self.aruco_dictionary = cv2.aruco.Dictionary_get(
            cv2.aruco.DICT_4X4_1000)
        self.enablePolygonsDisplay = self.args.enablePolygonsDisplay
        self.board_name = None
        if not self.args.debugProcessingMode:
            self.device = dai.Device()
            cameraProperties = self.device.getConnectedCameraFeatures()
            calibData = self.device.readCalibration()
            eeprom = calibData.getEepromData()
        #TODO Change only in getDeviceName in next revision.
        if self.args.board:
            self.board_name = self.args.board
            board_path = Path(Path(__file__).parent /self.args.board)
            if not board_path.exists():
                board_path = (Path(__file__).parent / 'resources/depthai_boards/boards' / self.args.board.upper()).with_suffix('.json').resolve()
                if not board_path.exists():
                    raise ValueError(
                        'Board config not found: {}'.format(Path(Path(__file__).parent /self.args.board)))
            with open(board_path) as fp:
                self.board_config = json.load(fp)
                self.board_config = self.board_config['board_config']
                self.board_config_backup = self.board_config
        elif not self.args.debugProcessingMode:
            try: 
                detection = self.device.getDeviceName()
                print(f"Device name: {detection}")
                detection = detection.split("-")
            except:
                cameraProperties = self.device.getConnectedCameraFeatures()
                calibData = self.device.readCalibration()
                eeprom = calibData.getEepromData()
                eeprom.productName = eeprom.productName.replace(" ", "-").upper()
                eeprom.boardName = eeprom.boardName.replace(" ", "-").upper()
                print(f"Product name: {eeprom.productName}, board name {eeprom.boardName}")
                if eeprom.productName.split("-")[0] == "OAK":
                    detection = eeprom.productName.split("-")
                elif eeprom.boardName.split("-")[0] == "OAK":
                    detection = eeprom.boardName.split("-")
                else:
                    raise ValueError(f"Board config for Product name: {eeprom.productName}, board name {eeprom.boardName} not found.")
            if "AF" in detection:
                detection.remove("AF")
            if "FF" in detection:
                detection.remove("FF")
            if "9782" in detection:
                detection.remove("9782")
            self.board_name = '-'.join(detection)
            board_path = Path(Path(__file__).parent /self.board_name)
            if self.traceLevel == 1:
                print(f"Board path specified as {board_path}")
            if not board_path.exists():
                board_path = (Path(__file__).parent / 'resources/depthai_boards/boards' / self.board_name.upper()).with_suffix('.json').resolve()
                if not board_path.exists():
                    raise ValueError(
                        'Board config not found: {}'.format(board_path))
            with open(board_path) as fp:
                self.board_config = json.load(fp)
                self.board_config = self.board_config['board_config']
                self.board_config_backup = self.board_config

        # TODO: set the total images
        # random polygons for count
        if self.args.numPictures:
            self.total_images = self.args.numPictures
        else:
            self.total_images = self.args.count * \
                len(calibUtils.setPolygonCoordinates(1000, 600))
        if self.traceLevel == 1:
            print("Using Arguments=", self.args)
        if self.args.datasetPath:
            path = Path(self.args.datasetPath).mkdir(parents=True, exist_ok=True)

        # if self.args.board.upper() == 'OAK-D-LITE':
        #     raise Exception(
        #     "OAK-D-Lite Calibration is not supported on main yet. Please use `lite_calibration` branch to calibrate your OAK-D-Lite!!")
        
        #TODO
        #if self.args.cameraMode != "perspective": 
            #self.args.minDetectedMarkersPercent = 1.0
        
        self.coverageImages ={} 
        for cam_id in self.board_config['cameras']:
            name = self.board_config['cameras'][cam_id]['name']
            self.coverageImages[name] = None
        if not self.args.debugProcessingMode:
            cameraProperties = self.device.getConnectedCameraFeatures()
            for properties in cameraProperties:
                for in_cam in self.board_config['cameras'].keys():
                    cam_info = self.board_config['cameras'][in_cam]
                    if cam_info["name"] not in self.args.disableCamera:
                        if properties.socket == stringToCam[in_cam]:
                            self.board_config['cameras'][in_cam]['sensorName'] = properties.sensorName
                            print('Cam: {} and focus: {}'.format(cam_info['name'], properties.hasAutofocus))
                            self.board_config['cameras'][in_cam]['hasAutofocus'] = properties.hasAutofocus
                            # self.auto_checkbox_dict[cam_info['name']  + '-Camera-connected'].check()
                            break

        self.charuco_board = cv2.aruco.CharucoBoard_create(
                            self.args.squaresX, self.args.squaresY,
                            self.args.squareSizeCm,
                            self.args.markerSizeCm,
                            self.aruco_dictionary)


    def mouse_event_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseTrigger = True

    def startPipeline(self):
        pipeline = self.create_pipeline()
        self.device.startPipeline(pipeline)

        self.camera_queue = {}
        for config_cam in self.board_config['cameras']:
            cam = self.board_config['cameras'][config_cam]
            if cam["name"] not in self.args.disableCamera:
                self.camera_queue[cam['name']] = self.device.getOutputQueue(cam['name'], 1, False)

    def is_markers_found(self, frame):
        marker_corners, _, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dictionary)
        print("Markers count ... {}".format(len(marker_corners)))
        num_all_markers = math.floor(self.args.squaresX * self.args.squaresY / 2)
        print(f'Total markers needed -> {int(num_all_markers * self.args.minDetectedMarkersPercent)}')
        return not (len(marker_corners) <  int(num_all_markers * self.args.minDetectedMarkersPercent))

    def detect_markers_corners(self, frame):
        marker_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.aruco_dictionary)
        marker_corners, ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(frame, self.charuco_board,
                                                                                marker_corners, ids, 
                                                                                rejectedCorners=rejectedImgPoints)
        if len(marker_corners) <= 0:
            return marker_corners, ids, None, None
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, ids, frame, self.charuco_board, minMarkers = 1)
        return marker_corners, ids, charuco_corners, charuco_ids

    def draw_markers(self, frame):
        marker_corners, ids, charuco_corners, charuco_ids = self.detect_markers_corners(frame)
        if charuco_ids is not None and len(charuco_ids) > 0:
            return cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
        return frame

    def draw_corners(self, frame, displayframe, color):
        marker_corners, ids, charuco_corners, charuco_ids = self.detect_markers_corners(frame)
        for corner in charuco_corners:
            corner_int = (int(corner[0][0]), int(corner[0][1]))
            cv2.circle(displayframe, corner_int, 8*displayframe.shape[1]//1900, color, -1)
        height, width = displayframe.shape[:2]
        start_point = (0, 0)  # top of the image
        end_point = (0, height)

        color = (0, 0, 0)  # blue in BGR
        thickness = 4

        # Draw the line on the image
        cv2.line(displayframe, start_point, end_point, color, thickness)
        return displayframe
        # return cv2.aruco.drawDetectedCornersCharuco(displayframe, charuco_corners)

    def test_camera_orientation(self, frame_l, frame_r):
        marker_corners_l, id_l, _ = cv2.aruco.detectMarkers(
            frame_l, self.aruco_dictionary)
        marker_corners_r, id_r, _ = cv2.aruco.detectMarkers(
            frame_r, self.aruco_dictionary)

        for i, left_id in enumerate(id_l):
            idx = np.where(id_r == left_id)
            # print(idx)
            if idx[0].size == 0:
                continue
            for left_corner, right_corner in zip(marker_corners_l[i], marker_corners_r[idx[0][0]]):
                if left_corner[0][0] - right_corner[0][0] < 0:
                    return False
        return True
    
    def create_pipeline(self):
        pipeline = dai.Pipeline()

        fps = self.args.framerate
        for cam_id in self.board_config['cameras']:
            cam_info = self.board_config['cameras'][cam_id]
            if cam_info["name"] not in self.args.disableCamera:
                if cam_info['type'] == 'mono':
                    cam_node = pipeline.createMonoCamera()
                    xout = pipeline.createXLinkOut()
                    sensorName = cam_info['sensorName']
                    print(f'Sensor name for {cam_info["name"]} is {sensorName}')
                    cam_node.setBoardSocket(stringToCam[cam_id])
                    cam_node.setResolution(camToMonoRes[cam_info['sensorName']])
                    cam_node.setFps(fps)

                    xout.setStreamName(cam_info['name'])
                    cam_node.out.link(xout.input)
                else:
                    cam_node = pipeline.createColorCamera()
                    xout = pipeline.createXLinkOut()

                    cam_node.setBoardSocket(stringToCam[cam_id])
                    sensorName = cam_info['sensorName']
                    print(f'Sensor name for {cam_info["name"]} is {sensorName}')
                    cam_node.setResolution(camToRgbRes[cam_info['sensorName'].upper()])
                    cam_node.setFps(fps)

                    xout.setStreamName(cam_info['name'])
                    cam_node.isp.link(xout.input)
                    if cam_info['sensorName'] == "OV9*82":
                        cam_node.initialControl.setSharpness(0)
                        cam_node.initialControl.setLumaDenoise(0)
                        cam_node.initialControl.setChromaDenoise(4)

                    if cam_info['hasAutofocus']:
                        if self.args.rgbLensPosition:
                            cam_node.initialControl.setManualFocus(int(self.args.rgbLensPosition[stringToCam[cam_id].name.lower()]))
                        else:
                            cam_node.initialControl.setManualFocus(135)

                        controlIn = pipeline.createXLinkIn()
                        controlIn.setStreamName(cam_info['name'] + '-control')
                        controlIn.out.link(cam_node.inputControl)

                cam_node.initialControl.setAntiBandingMode(antibandingOpts[self.args.antibanding])
                xout.input.setBlocking(False)
                xout.input.setQueueSize(1)

        return pipeline


    def parse_frame(self, frame, stream_name):
        if not self.is_markers_found(frame):
            return False

        filename = calibUtils.image_filename(self.current_polygon, self.images_captured)
        path = Path(self.args.datasetPath) / stream_name / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        print("py: Saved image as: " + str(path))
        return True

    def show_info_frame(self):
        info_frame = np.zeros((600, 1100, 3), np.uint8)
        print("Starting image capture. Press the [ESC] key to abort.")
        if self.enablePolygonsDisplay:
            print("Will take {} total images, {} per each polygon.".format(
                self.total_images, self.args.count))
        else:
            print("Will take {} total images.".format(
                self.total_images))

        def show(position, text):
            cv2.putText(info_frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
        show((25, 40), "Calibration of camera {}". format(self.board_name))
        show((25, 100), "Information about image capture:")
        show((25, 160), "Press the [ESC] key to abort.")
        show((25, 220), "Press the [spacebar] key to capture the image.")
        show((25, 280), "Press the \"s\" key to stop capturing images and begin calibration.")
        if self.enablePolygonsDisplay:
            show((25, 360), "Polygon on the image represents the desired chessboard")
            show((25, 420), "position, that will provide best calibration score.")
            show((25, 480), "Will take {} total images, {} per each polygon.".format(
                self.total_images, self.args.count))
        else:
            show((25, 480), "Will take {} total images.".format(
                self.total_images))
        show((25, 550), "To continue, press [spacebar]...")

        cv2.imshow("info", info_frame)
        while True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord(" "):
                cv2.destroyAllWindows()
                return
            elif key & 0xFF == 27 or key == ord("q"):  # 27 - ESC
                cv2.destroyAllWindows()
                raise SystemExit(0)

    def show_failed_capture_frame(self):
        width, height = int(
            self.width * self.output_scale_factor), int(self.height * self.output_scale_factor)
        info_frame = np.zeros((self.height, self.width, 3), np.uint8)
        if self.args.cameraMode != "perspective": 
            print("py: Capture failed, unable to find full board! Fix position and press spacebar again")
        else:
            print("py: Capture failed, unable to find chessboard! Fix position and press spacebar again")



        def show(position, text):
            cv2.putText(info_frame, text, position,
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0))

        show((50, int(height / 2 - 40)),
             "Capture failed, unable to find chessboard!")
        show((60, int(height / 2 + 40)), "Fix position and press spacebar again")

        # cv2.imshow("left", info_frame)
        # cv2.imshow("right", info_frame)
        cv2.imshow(self.display_name, info_frame)
        cv2.waitKey(1000)

    def show_failed_sync_images(self):
        width, height = int(
            self.width * self.output_scale_factor), int(self.height * self.output_scale_factor)
        info_frame = np.zeros((self.height, self.width, 3), np.uint8)
        print(f"py: Capture failed, unable to sync images! Fix the argument minSyncTimestamp or (-mts). Set to: {self.minSyncTimestamp}")
        def show(position, text):
            cv2.putText(info_frame, text, position,
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0))

        show((50, int(height / 2 - 40)),
             "Capture failed, unable to sync images!")
        show((60, int(height / 2 + 40)), "Fix the argument -mst.")

        # cv2.imshow("left", info_frame)
        # cv2.imshow("right", info_frame)
        cv2.imshow(self.display_name, info_frame)
        cv2.waitKey(0)

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


    def empty_calibration(self, calib: dai.CalibrationHandler):
        data = calib.getEepromData()
        for attr in ["boardName", "boardRev"]:
            if getattr(data, attr): return False
        return True
    
    def capture_images_sync(self):
        finished = False
        capturing = False
        start_timer = False
        timer = self.args.captureDelay
        prev_time = None
        curr_time = None

        self.display_name = "Image Window"
        self.minSyncTimestamp = self.args.minSyncTimestamp
        syncCollector = MessageSync(len(self.camera_queue), self.minSyncTimestamp) # 3ms tolerance
        syncCollector.traceLevel = self.args.traceLevel
        self.mouseTrigger = False
        sync_trys = 0
        while not finished:
            currImageList = {}
            for key in self.camera_queue.keys():
                frameMsg = self.camera_queue[key].get()

                #print(f'Timestamp of  {key} is {frameMsg.getTimestamp()}')

                syncCollector.add_msg(key, frameMsg)
                color_frame = None
                if frameMsg.getType() in [dai.RawImgFrame.Type.RAW8, dai.RawImgFrame.Type.GRAY8] :
                    color_frame = cv2.cvtColor(frameMsg.getCvFrame(), cv2.COLOR_GRAY2BGR)
                else:
                    color_frame = frameMsg.getCvFrame()
                currImageList[key] = color_frame
                # print(gray_frame.shape)

            resizeHeight = 0
            resizeWidth = 0
            for name, imgFrame in currImageList.items():
                #self.coverageImages[name]=None
                
                # print(f'original Shape of {name} is {imgFrame.shape}' )
               
                currImageList[name] = cv2.resize(self.draw_markers(imgFrame),
                                                 (0, 0), 
                                                 fx=self.output_scale_factor, 
                                                 fy=self.output_scale_factor)
                
                height, width, _ = currImageList[name].shape

                widthRatio = resizeWidth / width
                heightRatio = resizeHeight / height


                if (widthRatio > 0.8 and heightRatio > 0.8 and widthRatio <= 1.0 and heightRatio <= 1.0) or (widthRatio > 1.2 and heightRatio > 1.2) or (resizeHeight == 0):
                    resizeWidth = width
                    resizeHeight = height
                # elif widthRatio > 1.2 and heightRatio > 1.2:


                # if width < resizeWidth:
                #     resizeWidth = width
                # if height > resizeHeight:
                #     resizeHeight = height
            
            # print(f'Scale Shape  is {resizeWidth}x{resizeHeight}' )
                if self.args.invert_v and self.args.invert_h:
                    currImageList[name] = cv2.flip(currImageList[name], -1)
                elif self.args.invert_v:
                    currImageList[name] = cv2.flip(currImageList[name], 0)
                elif self.args.invert_h:
                    currImageList[name] = cv2.flip(currImageList[name], 1)
            
            combinedImage = None
            combinedCoverageImage = None
            for name, imgFrame in currImageList.items():
                height, width, _ = imgFrame.shape
                if width > resizeWidth and height > resizeHeight:
                    imgFrame = cv2.resize(
                    imgFrame, (0, 0), fx= resizeWidth / width, fy= resizeWidth / width)
                
                # print(f'final_scaledImageSize is {imgFrame.shape}')
                if self.polygons is None:
                    self.height, self.width, _ = imgFrame.shape
                    # print(self.height, self.width)
                    self.polygons = calibUtils.setPolygonCoordinates(
                        self.height, self.width)
                if self.current_polygon<len(self.polygons):
                    localPolygon = np.array([self.polygons[self.current_polygon]])
                # print(localPolygon.shape)
                # print(localPolygon)
                if self.images_captured_polygon == 1:
                    # perspectiveRotationMatrix = Rotation.from_euler('z', 45, degrees=True).as_matrix()
                    angle = 30.
                    theta = (angle/180.) * np.pi
                    perspectiveRotationMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                                        [np.sin(theta),  np.cos(theta)]])
                    
                    localPolygon = np.matmul(localPolygon, perspectiveRotationMatrix).astype(np.int32)
                    localPolygon[0][:, 1] += abs(localPolygon.min())    
                if self.images_captured_polygon == 2:
                    # perspectiveRotationMatrix = Rotation.from_euler('z', -45, degrees=True).as_matrix()
                    angle = -30.
                    theta = (angle/180.) * np.pi
                    perspectiveRotationMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                                        [np.sin(theta),  np.cos(theta)]])
                    localPolygon = np.matmul(localPolygon, perspectiveRotationMatrix).astype(np.int32)
                    localPolygon[0][:, 1] += (height - abs(localPolygon[0][:, 1].max()))    
                    localPolygon[0][:, 0] += abs(localPolygon[0][:, 1].min())    
                if self.images_captured_polygon<len(self.polygons) and self.args.enablePolygonsDisplay:
                    cv2.polylines(
                        imgFrame, localPolygon,
                        True, (0, 0, 255), 4)
                
                height, width, _ = imgFrame.shape
                # TO-DO: fix the rooquick and dirty fix: if the resized image is higher than the target resolution, crop it
                if height > resizeHeight:
                    height_offset = (height - resizeHeight)//2
                    imgFrame = imgFrame[height_offset:height_offset+resizeHeight, :]

                height, width, _ = imgFrame.shape
                height_offset = (resizeHeight - height)//2
                width_offset = (resizeWidth - width)//2
                subImage = np.pad(imgFrame, ((height_offset, height_offset), (width_offset, width_offset), (0, 0)), 'constant', constant_values=0)
                if self.coverageImages[name] is not None:
                    if len(self.coverageImages[name].shape) != 3:
                        self.coverageImages[name] = cv2.cvtColor(self.coverageImages[name], cv2.COLOR_GRAY2RGB)
                    imgFrame = self.coverageImages[name]
                    cv2.resize(imgFrame, (0, 0), fx=self.output_scale_factor*2, fy=self.output_scale_factor*2)
                    height, width, _ = imgFrame.shape
                    if width > resizeWidth and height > resizeHeight:
                        imgFrame = cv2.resize(
                        imgFrame, (0, 0), fx= resizeWidth / width, fy= resizeWidth / width)
                    height, width, _ = imgFrame.shape
                    if height > resizeHeight:
                        height_offset = (height - resizeHeight)//2
                        imgFrame = imgFrame[height_offset:height_offset+resizeHeight, :]
                    height, width, _ = imgFrame.shape
                    height_offset = (resizeHeight - height)//2
                    width_offset = (resizeWidth - width)//2
                    padding = ((height_offset, height_offset), (width_offset,width_offset), (0, 0))
                    subCoverageImage = np.pad(imgFrame, padding, 'constant', constant_values=0)
                    print_text = f"Camera: {name}, picture {self.images_captured}"
                    cv2.putText(subCoverageImage, print_text, (15, 15+height_offset), cv2.FONT_HERSHEY_SIMPLEX, 2*imgFrame.shape[0]/1750, (0, 0, 0), 2)                    
                    if combinedCoverageImage is None:
                        combinedCoverageImage = subCoverageImage
                    else:
                        combinedCoverageImage = np.hstack((combinedCoverageImage, subCoverageImage))
                if combinedImage is None:
                    combinedImage = subImage
                else:
                    combinedImage = np.hstack((combinedImage, subImage))
                
            key = cv2.waitKey(1)
            if (key & 0xFF)== 27  or (key & 0xFF) == ord("q"):
                print("py: Calibration has been interrupted!")
                raise SystemExit(0)
            elif key == ord(" ") or self.mouseTrigger == True:
                start_timer = True
                prev_time = time.time()
                timer = self.args.captureDelay
                self.mouseTrigger = False
            if key == 27 or key == ord("s"):
                finished = True
                print("Capturing interrupted by user, procceding with processing of images.")
                cv2.destroyAllWindows()
                break

            display_image = combinedImage
            if start_timer == True:
                curr_time = time.time()
                if curr_time - prev_time >= 1:
                    prev_time = curr_time
                    timer = timer - 1
                if timer <= 0 and start_timer == True:
                    start_timer = False
                    capturing = True
                    print('Start capturing...')
                
                image_shape = combinedImage.shape

                cv2.putText(display_image, str(timer),
                        (image_shape[1]//2, image_shape[0]//2), font,
                        7, (0, 0, 255),
                        4, cv2.LINE_AA)
            cv2.namedWindow(self.display_name)
            if self.args.mouseTrigger:
                cv2.setMouseCallback(self.display_name, self.mouse_event_callback)

            cv2.imshow(self.display_name, display_image)
            if combinedCoverageImage is not None:
                #combinedCoverageImage = cv2.resize(combinedCoverageImage, (0, 0), fx=self.output_scale_factor*2, fy=self.output_scale_factor*2)
                cv2.imshow("Coverage-Image", combinedCoverageImage)

            tried = {}
            allPassed = True
            if capturing:
                syncedMsgs = syncCollector.get_synced()
                if sync_trys > 10:
                    self.show_failed_sync_images()
                    finished = True
                    self.device.close()
                    print("Images were unable to sync, threshold to high. Device closing with exception.")
                    raise SystemExit(1)
                if syncedMsgs == False or syncedMsgs == None:
                    for key in self.camera_queue.keys():
                        self.camera_queue[key].getAll()
                    sync_trys += 1
                    continue

                for name, frameMsg in syncedMsgs.items():
                    print(f"Time stamp of {name} is {frameMsg.getTimestamp()}")
                    if self.coverageImages[name] is None:
                        coverageShape = frameMsg.getCvFrame().shape
                        self.coverageImages[name] = np.ones(coverageShape, np.uint8) * 255

                    tried[name] = self.parse_frame(frameMsg.getCvFrame(), name)
                    print(f'Status of {name} is {tried[name]}')
                    allPassed = allPassed and tried[name]
                if allPassed:
                    color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
                    for name, frameMsg in syncedMsgs.items():
                        frameMsg_frame = frameMsg.getCvFrame()
                        if len(frameMsg.getCvFrame().shape) != 3:
                            frameMsg_frame = cv2.cvtColor(frameMsg.getCvFrame(), cv2.COLOR_GRAY2RGB)
                        if len(self.coverageImages[name].shape) != 3:
                            self.coverageImages[name] = cv2.cvtColor(self.coverageImages[name], cv2.COLOR_GRAY2RGB)
                        self.coverageImages[name] = self.draw_corners(frameMsg_frame, self.coverageImages[name], color)
                    if not self.images_captured:
                        if 'stereo_config' in self.board_config['cameras']:
                            leftStereo =  self.board_config['cameras'][self.board_config['stereo_config']['left_cam']]['name']
                            rightStereo = self.board_config['cameras'][self.board_config['stereo_config']['right_cam']]['name']
                            print(f'Left Camera of stereo is {leftStereo} and right Camera of stereo is {rightStereo}')
                        # if not self.test_camera_orientation(syncedMsgs[leftStereo].getCvFrame(), syncedMsgs[rightStereo].getCvFrame()):
                        #     self.show_failed_orientation()

                    self.images_captured += 1
                    self.images_captured_polygon += 1
                    capturing = False
                else:
                    self.show_failed_capture_frame()
                    capturing = False
            # print(f'self.images_captured_polygon  {self.images_captured_polygon}')
            # print(f'self.current_polygon  {self.current_polygon}')
            # print(f'len(self.polygons)  {len(self.polygons)}')

            if self.images_captured_polygon == self.args.count:
                self.images_captured_polygon = 0
                self.current_polygon += 1
                if self.args.numPictures == None:
                    if self.current_polygon == len(self.polygons):
                        finished = True
                        cv2.destroyAllWindows()
                        break
                else:
                    if self.images_captured == self.args.numPictures:
                        finished = True
                        cv2.destroyAllWindows()
                        break




    def calibrate(self):
        print("Starting image processing")
        stereo_calib = calibUtils.StereoCalibration(self.args.traceLevel, self.args.outputScaleFactor, self.args.disableCamera)
        dest_path = str(Path('resources').absolute())
        # self.args.cameraMode = 'perspective' # hardcoded for now
        try:
            # stereo_calib = StereoCalibration()
            status, result_config = stereo_calib.calibrate(
                                        self.board_config,
                                        self.dataset_path,
                                        self.args.squareSizeCm,
                                        self.args.markerSizeCm,
                                        self.args.squaresX,
                                        self.args.squaresY,
                                        self.args.cameraMode,
                                        self.args.rectifiedDisp) # Turn off enable disp rectify

            if self.args.noInitCalibration or self.args.debugProcessingMode:
                calibration_handler = dai.CalibrationHandler()
            else:
                calibration_handler = self.device.readCalibration()
            board_keys = self.board_config.keys()
            try:
                if self.empty_calibration(calibration_handler):
                    if "name" in board_keys and "revision" in board_keys:
                        calibration_handler.setBoardInfo(self.board_config['name'], self.board_config['revision'])
                    else:
                        calibration_handler.setBoardInfo(str(self.device.getDeviceName()), str(self.args.revision))
            except Exception as e:
                print('Device closed in exception..' )
                if not self.args.debugProcessingMode:
                    self.device.close()
                print(e)
                print(traceback.format_exc())
                raise SystemExit(1)
            
            target_file = open(self.dataset_path + '/target_info.txt', 'w')
            # calibration_handler.set
            error_text = []
            
            target_file.write(f'Marker Size: {self.args.markerSizeCm} cm\n')
            target_file.write(f'Square Size: {self.args.squareSizeCm} cm\n')
            target_file.write(f'Number of squaresX: {self.args.squaresX}\n')
            target_file.write(f'Number of squaresY: {self.args.squaresY}\n')

            for camera in result_config['cameras'].keys():
                cam_info = result_config['cameras'][camera]
                if cam_info["name"] not in self.args.disableCamera:
                    # log_list.append(self.ccm_selected[cam_info['name']])
                    reprojection_error_threshold = 1.0
                    if cam_info['size'][1] > 720:
                        #print(cam_info['size'][1])
                        reprojection_error_threshold = reprojection_error_threshold * cam_info['size'][1] / 720

                    if cam_info['name'] == 'rgb':
                        reprojection_error_threshold = 3
                    print('Reprojection error threshold -> {}'.format(reprojection_error_threshold))

                    if cam_info['reprojection_error'] > reprojection_error_threshold:
                        color = red
                        error_text.append("high Reprojection Error")
                    text = cam_info['name'] + ' Reprojection Error: ' + format(cam_info['reprojection_error'], '.6f')
                    print(text)
                    text = cam_info['name'] + '-reprojection: {}\n'.format(cam_info['reprojection_error'], '.6f')
                    target_file.write(text)

                    # pygame_render_text(self.screen, text, (vis_x, vis_y), color, 30)

                    calibration_handler.setDistortionCoefficients(stringToCam[camera], cam_info['dist_coeff'])
                    calibration_handler.setCameraIntrinsics(stringToCam[camera], cam_info['intrinsics'],  cam_info['size'][0], cam_info['size'][1])
                    calibration_handler.setFov(stringToCam[camera], cam_info['hfov'])
                    if self.args.cameraMode != 'perspective':
                        calibration_handler.setCameraType(stringToCam[camera], dai.CameraModel.Fisheye)
                    if 'hasAutofocus' in cam_info and cam_info['hasAutofocus']:
                        if self.args.rgbLensPosition:
                            calibration_handler.setLensPosition(stringToCam[camera], int(self.args.rgbLensPosition[cam_info["name"]]))
                        else:
                            calibration_handler.setLensPosition(stringToCam[camera], int(135))

                    # log_list.append(self.focusSigma[cam_info['name']])
                    # log_list.append(cam_info['reprojection_error'])
                    # color = green///
                    # epErrorZText 
                    if 'extrinsics' in cam_info:
                        if 'to_cam' in cam_info['extrinsics']:
                            right_cam = result_config['cameras'][cam_info['extrinsics']['to_cam']]['name']
                            if right_cam not in self.args.disableCamera:
                                left_cam = cam_info['name']

                                epipolar_threshold = self.args.maxEpiploarError

                                if cam_info['extrinsics']['epipolar_error'] > epipolar_threshold:
                                    color = red
                                    error_text.append("high epipolar error between " + left_cam + " and " + right_cam)
                                elif cam_info['extrinsics']['epipolar_error'] == -1:
                                    color = red
                                    error_text.append("Epiploar validation failed between " + left_cam + " and " + right_cam)

                                text = cam_info['name'] + " and " + right_cam + ' epipolar_error: {}\n'.format(cam_info['extrinsics']['epipolar_error'], '.6f')
                                target_file.write(text)

                                # log_list.append(cam_info['extrinsics']['epipolar_error'])
                                # text = left_cam + "-" + right_cam + ' Avg Epipolar error: ' + format(cam_info['extrinsics']['epipolar_error'], '.6f')
                                # pygame_render_text(self.screen, text, (vis_x, vis_y), color, 30)
                                # vis_y += 30
                                specTranslation = np.array([cam_info['extrinsics']['specTranslation']['x'], cam_info['extrinsics']['specTranslation']['y'], cam_info['extrinsics']['specTranslation']['z']], dtype=np.float32)

                                calibration_handler.setCameraExtrinsics(stringToCam[camera], stringToCam[cam_info['extrinsics']['to_cam']], cam_info['extrinsics']['rotation_matrix'], cam_info['extrinsics']['translation'], specTranslation)
                                if result_config['stereo_config']['left_cam'] == camera and result_config['stereo_config']['right_cam'] == cam_info['extrinsics']['to_cam']:
                                    calibration_handler.setStereoLeft(stringToCam[camera], result_config['stereo_config']['rectification_left'])
                                    calibration_handler.setStereoRight(stringToCam[cam_info['extrinsics']['to_cam']], result_config['stereo_config']['rectification_right'])
                                elif result_config['stereo_config']['left_cam'] == cam_info['extrinsics']['to_cam'] and result_config['stereo_config']['right_cam'] == camera:                           
                                    calibration_handler.setStereoRight(stringToCam[camera], result_config['stereo_config']['rectification_right'])
                                    calibration_handler.setStereoLeft(stringToCam[cam_info['extrinsics']['to_cam']], result_config['stereo_config']['rectification_left'])
            target_file.close()

            if len(error_text) == 0 and not self.args.debugProcessingMode:
                print('Flashing Calibration data into ')
                # print(calib_dest_path)

                eeepromData = calibration_handler.getEepromData()
                print(f'EEPROM VERSION being flashed is  -> {eeepromData.version}')
                eeepromData.version = 7
                print(f'EEPROM VERSION being flashed is  -> {eeepromData.version}')
                mx_serial_id = self.device.getDeviceInfo().getMxId()
                date_time_string = datetime.now().strftime("_%m_%d_%y_%H_%M")
                file_name = mx_serial_id + date_time_string
                calib_dest_path = dest_path + '/' + file_name + '.json'
                calibration_handler.eepromToJsonFile(calib_dest_path)
                if self.args.saveCalibPath:
                    Path(self.args.saveCalibPath).parent.mkdir(parents=True, exist_ok=True)
                    calibration_handler.eepromToJsonFile(self.args.saveCalibPath)
                # try:
                self.device.flashCalibration2(calibration_handler)
                is_write_succesful = True
                # except RuntimeError as e:
                #     is_write_succesful = False
                #     print(e)
                #     print("Writing in except...")
                #     is_write_succesful = self.device.flashCalibration2(calibration_handler)

                if self.args.factoryCalibration:
                    try:
                        self.device.flashFactoryCalibration(calibration_handler)
                        is_write_factory_sucessful = True
                    except RuntimeError:
                        print("flashFactoryCalibration Failed...")
                        is_write_factory_sucessful = False

                if is_write_succesful:
                    if not self.args.debugProcessingMode:
                        self.device.close()
                    text = "EEPROM written succesfully"
                    resImage = create_blank(900, 512, rgb_color=green)
                    cv2.putText(resImage, text, (10, 250), font, 2, (0, 0, 0), 2)
                    cv2.imshow("Result Image", resImage)
                    cv2.waitKey(0)
                    
                else:
                    if not self.args.debugProcessingMode:
                        self.device.close()
                    text = "EEPROM write Failed!!"
                    resImage = create_blank(900, 512, rgb_color=red)
                    cv2.putText(resImage, text, (10, 250), font, 2, (0, 0, 0), 2)
                    cv2.imshow("Result Image", resImage)
                    cv2.waitKey(0)
                    # return (False, "EEPROM write Failed!!")
            
            else:
                if not self.args.debugProcessingMode:
                    self.device.close()
                print(error_text)
                for text in error_text: 
                # text = error_text[0]                
                    resImage = create_blank(900, 512, rgb_color=red)
                    cv2.putText(resImage, text, (10, 250), font, 2, (0, 0, 0), 2)
                    cv2.imshow("Result Image", resImage)
                    cv2.waitKey(0)
        except Exception as e:
            if not self.args.debugProcessingMode:
                self.device.close()
            print('Device closed in exception..' )
            print(e)
            print(traceback.format_exc())
            raise SystemExit(1)

    def run(self):
        if 'capture' in self.args.mode:
            if self.args.datasetPath:
                print("Saving dataset to: {}".format(self.args.datasetPath))
                self.dataset_path = self.args.datasetPath
                if self.dataset_path != "dataset":
                    answer = input("This folders content will be deleted for sake of calibration: Proceed? (y/n)")
                    if answer == "y" or answer == "Y":
                        print("Starting calibration")
                    else:
                        print("Calibration ended by user.")
                        raise SystemExit(1)
            try:
                if Path(self.dataset_path).exists():
                    shutil.rmtree(Path(self.dataset_path))
                for cam_id in self.board_config['cameras']:
                    name = self.board_config['cameras'][cam_id]['name']
                    Path(Path(self.dataset_path) / "{}".format(name)).mkdir(parents=True, exist_ok=True)
                
            except OSError:
                traceback.print_exc()
                print("An error occurred trying to create image dataset directories!")
                raise SystemExit(1)
            self.startPipeline()
            self.show_info_frame()
            self.capture_images_sync()
        print("Using dataset path: {}".format(self.args.datasetPath))
        self.dataset_path = self.args.datasetPath
        if 'process' in self.args.mode:
            self.calibrate()
        print('py: DONE.')


if __name__ == "__main__":
    Main().run()

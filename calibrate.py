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
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--polygons", default=list(np.arange(len(setPolygonCoordinates(1000,600)))), nargs='*',
                        type=int, required=False,
                        help="Space-separated list of polygons (ex: 0 5 7) to restrict image capture. Default is all polygons.")
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

    options = parser.parse_args()

    return options

args = vars(parse_args())

def find_chessboard(frame, small=True):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if small:
        frame = cv2.resize(frame, (0, 0), fx = 0.3, fy = 0.3)
    board_detected, corners = cv2.findChessboardCorners(frame, (9,6),chessboard_flags)
    return  board_detected



if args['config_overwrite']:
    args['config_overwrite'] = json.loads(args['config_overwrite'])

print("Using Arguments=",args)

if 'capture' in args['mode']:

    # Delete Dataset directory if asked
    if args['image_op'] == 'delete':
        shutil.rmtree('dataset/')

    # Creates dirs to save captured images
    try:
        for path in ["left","right"]:
            Path("dataset/"+path).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print ("An error occurred trying to create image dataset directories:",e)
        exit(0)

    # Create Depth AI Pipeline to start video streaming
    cmd_file = consts.resource_paths.device_cmd_fpath

    # Possible to try and reboot?
    # The following doesn't work (have to manually hit switch on device)
    # depthai.reboot_device
    # time.sleep(1)
    if not depthai.init_device(cmd_file=cmd_file):
        print("[ERROR] Unable to initialize device. Try to reset it. Exiting.")
        exit(1)

    config = {
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
            'swap_left_and_right_cameras': True,
            'left_fov_deg': 69.0,
            'left_to_right_distance_cm': 9.0
        }
    }

    if args['config_overwrite'] is not None:
        config = utils.merge(args['config_overwrite'],config)
        print("Merged Pipeline config with overwrite",config)

    pipeline = depthai.create_pipeline(config)

    if pipeline is None:
        print("[ERROR] Unable to create pipeline. Exiting.")
        exit(2)

    num_of_polygons = 0
    polygons_coordinates = []

    image_per_polygon_counter = 0 # variable to track how much images were captured per each polygon
    complete = False # Indicates if images have been captured for all polygons

    polygon_index = args['polygons'][0] # number to track which polygon is currently using
    total_num_of_captured_images = 0 # variable to hold total number of captured images

    capture_images = False # value to track the state of capture button (spacebar)
    captured_left_image = False # value to check if image from the left camera was capture
    captured_right_image = False # value to check if image from the right camera was capture

    run_capturing_images = True # value becames False and stop the main loop when all polygon indexes were used

    calculate_coordinates = False # track if coordinates of polynoms was calculated
    total_images = args['count']*len(args['polygons'])

    # Chessboard detection termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    framecnt = 0
    leftcolor = (0, 0, 255)
    rightcolor = (0, 0, 255)

    while run_capturing_images:
        _, data_list = pipeline.get_available_nnet_and_data_packets()
        for packet in data_list:
            if packet.stream_name == 'left' or packet.stream_name == 'right':
                framecnt += 1            
                frame = packet.getData()

                if calculate_coordinates == False:
                    height, width = frame.shape
                    polygons_coordinates = setPolygonCoordinates(height, width)
                    # polygons_coordinates = select_polygon_coords(polygons_coordinates,args['polygons'])
                    num_of_polygons = len(args['polygons'])
                    print("Starting image capture. Press the [ESC] key to abort.")
                    print("Will take %i total images, %i per each polygon." % (total_images,args['count']))
                    calculate_coordinates = True

                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                if capture_images == True:
                    if packet.stream_name == 'left':
                        if find_chessboard(frame, False):
                            filename = image_filename(packet.stream_name,polygon_index,total_num_of_captured_images)
                            cv2.imwrite("dataset/left/" + str(filename), frame)
                            print("py: Saved image as: " + str(filename))
                            captured_left_image = True
                        else:
                            print("py: could not find chessboard, try again")
                            capture_images, captured_left_image, captured_right_image = False, False, False


                    elif packet.stream_name == 'right':
                        if find_chessboard(frame, False):
                            filename = image_filename(packet.stream_name,polygon_index,total_num_of_captured_images)
                            cv2.imwrite("dataset/right/" + str(filename), frame)
                            print("py: Saved image as: " + str(filename))
                            captured_right_image = True
                        else:
                            print("py: could not find chess board, try again")
                            capture_images, captured_left_image, captured_right_image = False, False, False

                    if captured_right_image == True and captured_left_image == True:
                        capture_images = False
                        captured_left_image = False
                        captured_right_image = False
                        total_num_of_captured_images += 1
                        image_per_polygon_counter += 1

                        if image_per_polygon_counter == args['count']:
                            image_per_polygon_counter = 0
                            try:
                                polygon_index = args['polygons'][args['polygons'].index(polygon_index)+1]
                            except IndexError:
                                complete = True

                if complete == False:                    
                    if framecnt % 60 is 0:
                        # Find the chess board corners once a second
                        if find_chessboard(frame):
                            rightcolor = (0, 255, 0)
                        else:
                            rightcolor = (0, 0, 255)
                    if framecnt % 61 is 0:
                        if find_chessboard(frame):
                            leftcolor = (0, 255, 0)
                        else:
                            leftcolor = (0, 0, 255)

                    cv2.putText(frame, "Align cameras with callibration board and press spacebar to capture the image:", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                    cv2.putText(frame, "Polygon Position: %i. " % (polygon_index) + "Captured %i of %i images." % (total_num_of_captured_images,total_images), (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

                    if packet.stream_name == 'left':
                        cv2.polylines(frame, np.array([getPolygonCoordinates(polygon_index, polygons_coordinates)]), True, leftcolor , 4)
                    else:
                        cv2.polylines(frame, np.array([getPolygonCoordinates(polygon_index, polygons_coordinates)]), True, rightcolor , 4)



                    frame = cv2.resize(frame, (0, 0), fx = 0.8, fy = 0.8)
                    cv2.imshow(packet.stream_name, frame)

                else:
                    # all polygons used, stop the loop
                    run_capturing_images = False




        # key = cv2.waitKey(33)
        #
        # if key == 27: # ESC is consistent w/exiting stereo image preview.
        #     capture_images = True
        #
        # elif key == ord("q"):
        #     print("py: Calibration has been interrupted!")
        #     exit(0)

        key = cv2.waitKey(1)

        if key == ord(" "):
            capture_images = True

        elif key == ord("q"):
            print("py: Calibration has been interrupted!")
            exit(0)


    del pipeline # need to manualy delete the object, because of size of HostDataPacket queue runs out (Not enough free space to save {stream})

    cv2.destroyWindow("left")
    cv2.destroyWindow("right")

else:
    print("Skipping capture.")

if 'process' in args['mode']:
    print("Starting image processing")
    cal_data = StereoCalibration()
    try:
        cal_data.calibrate("dataset", args['square_size_cm'], "./resources/depthai.calib")
    except AssertionError as e:
        print("[ERROR] " + str(e))
        exit(1)
else:
    print("Skipping process.")

print('py: DONE.')

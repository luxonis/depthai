import depthai
from  calibration_utils import *
from argparse import ArgumentParser
from time import time
import numpy as np
import os
from pathlib import Path
import shutil
import consts.resource_paths

use_cv = True
try:
    import cv2
except ImportError:
    use_cv = False

def parse_args():
    parser = ArgumentParser()
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
    options = parser.parse_args()

    return options

args = vars(parse_args())
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

    cmd_file = consts.resource_paths.device_depth_cmd_fpath

    # Create Depth AI Pipeline to start video streaming
    streams_list = ['left', 'right', 'depth']
    pipieline = depthai.create_pipeline(
            streams=streams_list,
            cmd_file=cmd_file,
            calibration_file=consts.resource_paths.calib_fpath,
            config_file=consts.resource_paths.pipeline_config_fpath
            )

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

    while run_capturing_images:
        data_list = pipieline.get_available_data_packets()
        for packet in data_list:
            if packet.stream_name == 'left' or packet.stream_name == 'right':
                frame = packet.getData()
                if calculate_coordinates == False:
                    height, width = frame.shape
                    polygons_coordinates = setPolygonCoordinates(height, width)
                    # polygons_coordinates = select_polygon_coords(polygons_coordinates,args['polygons'])
                    num_of_polygons = len(args['polygons'])
                    print("Will take %i total images, %i per each polygon." % (total_images,args['count']))
                    calculate_coordinates = True

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if capture_images == True:
                    if packet.stream_name == 'left':
                        filename = image_filename(packet.stream_name,polygon_index,total_num_of_captured_images)
                        cv2.imwrite("dataset/left/" + str(filename), frame)
                        print("py: Saved image as: " + str(filename))
                        captured_left_image = True

                    elif packet.stream_name == 'right':
                        filename = image_filename(packet.stream_name,polygon_index,total_num_of_captured_images)
                        cv2.imwrite("dataset/right/" + str(filename), frame)
                        print("py: Saved image as: " + str(filename))
                        captured_right_image = True

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
                    cv2.putText(frame, "Align cameras with callibration board and press spacebar to capture the image", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                    cv2.putText(frame, "Polygon Position: %i. " % (polygon_index) + "Captured %i of %i images." % (total_num_of_captured_images,total_images), (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                    cv2.polylines(frame, np.array([getPolygonCoordinates(polygon_index, polygons_coordinates)]), True, (0, 0, 255), 4)
                    # original image is 1280x720. reduce by 2x so it fits better.
                    aspect_ratio = 1.5
                    new_x, new_y = int(frame.shape[1]/aspect_ratio), int(frame.shape[0]/aspect_ratio)
                    resized_image = cv2.resize(frame,(new_x,new_y))
                    cv2.imshow(packet.stream_name, resized_image)
                else:
                    # all polygons used, stop the loop
                    run_capturing_images = False

        key = cv2.waitKey(1)

        if key == ord(" "):
            capture_images = True

        elif key == ord("q"):
            print("py: Calibration has been interrupted!")
            exit(0)


    del pipieline # need to manualy delete the object, because of size of HostDataPacket queue runs out (Not enough free space to save {stream})

    cv2.destroyWindow("left")
    cv2.destroyWindow("right")

else:
    print("Skipping capture.")

if 'process' in args['mode']:
    print("Starting image processing")
    cal_data = StereoCalibration()
    try:
        cal_data.calibrate("dataset", args['square_size_cm'], "./depthai.calib")
    except AssertionError as e:
        print("[ERROR] " + str(e))
        exit(0)
else:
    print("Skipping process.")

print('py: DONE.')

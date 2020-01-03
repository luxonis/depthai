import depthai
import calibration_utils

from time import time
import numpy as np
import os

use_cv = True
try:
    import cv2
except ImportError:
    use_cv = False


def setPolygonCoordinates(height, width):
    horizontal_shift = width//4
    vertical_shift = height//4

    margin = 60
    slope = 150

    p_coordinates = [
            [[margin,0], [margin,height], [width//2, height-slope], [width//2, slope]],
            [[horizontal_shift, 0], [horizontal_shift, height], [width//2 + horizontal_shift, height-slope], [width//2 + horizontal_shift, slope]],
            [[horizontal_shift*2-margin, 0], [horizontal_shift*2-margin, height], [width//2 + horizontal_shift*2-margin, height-slope], [width//2 + horizontal_shift*2-margin, slope]],

            [[margin,margin], [margin, height-margin], [width-margin, height-margin], [width-margin, margin]],

            [[width-margin, 0], [width-margin, height], [width//2, height-slope], [width//2, slope]],
            [[width-horizontal_shift, 0], [width-horizontal_shift, height], [width//2-horizontal_shift, height-slope], [width//2-horizontal_shift, slope]],
            [[width-horizontal_shift*2+margin, 0], [width-horizontal_shift*2+margin, height], [width//2-horizontal_shift*2+margin, height-slope], [width//2-horizontal_shift*2+margin, slope]],

            [[0,margin], [width, margin], [width-slope, height//2], [slope, height//2]],
            [[0,vertical_shift], [width, vertical_shift], [width-slope, height//2+vertical_shift], [slope, height//2+vertical_shift]],
            [[0,vertical_shift*2-margin], [width, vertical_shift*2-margin], [width-slope, height//2+vertical_shift*2-margin], [slope, height//2+vertical_shift*2-margin]],

            [[0,height-margin], [width, height-margin], [width-slope, height//2], [slope, height//2]],
            [[0,height-vertical_shift], [width, height-vertical_shift], [width-slope, height//2-vertical_shift], [slope, height//2-vertical_shift]],
            [[0,height-vertical_shift*2+margin], [width, height-vertical_shift*2+margin], [width-slope, height//2-vertical_shift*2+margin], [slope, height//2-vertical_shift*2+margin]]
        ]
    return p_coordinates

def getPolygonCoordinates(idx, p_coordinates):
    return p_coordinates[idx]

def getNumOfPolygons(p_coordinates):
    return len(p_coordinates)



# creates dirs to save captured images
try:
    os.mkdir("dataset")
    os.chdir("dataset")
    os.mkdir("left")
    os.mkdir("right")
    os.chdir("../")
except OSError:
    print ("py: Creation of the directories for images from left and right cameras have failed")
    exit(0)

# get size of calibration pattern, uses to calibrate cameras
calib_puttern_size = input("Enter size of calibration pattern used, Please use [cm] (Default value: 2.5 cm): ")

if calib_puttern_size == "":
    calib_puttern_size = 2.5
else:
    calib_puttern_size = float(calib_puttern_size)

print("py: Size of calibration = " + str(calib_puttern_size))


streams_list = ['left', 'right', 'depth', 'meta_d2h']
pipieline = depthai.create_pipeline(
        streams=streams_list,
        cmd_file='./depthai_depth.cmd'
        )

num_of_polygons = 0
polygons_coordinates = []

num_image_per_polygon = 5 # number of images captured per polygon, this value can be changed, the bigger value, the more images per each polygon will be captured
image_per_polygon_counter = 0 # variable to track how much images were captured per each polygon

polygon_index = 0 # number to track which polygon is currently using
total_num_of_captured_images = 0 # variable to hold total number of captured images

capture_images = False # value to track the state of capture button (spacebar)
captured_left_image = False # value to check if image from the left camera was capture
captured_right_image = False # value to check if image from the right camera was capture

run_capturing_images = True # value becames False and stop the main loop when all polygon indexes were used

calculate_coordinates = False # track if coordinates of polynoms was calculated

while run_capturing_images:
    data_list = pipieline.get_available_data_packets()
    for packet in data_list:
        if packet.stream_name == 'left' or packet.stream_name == 'right':
            frame = packet.getData()
            if calculate_coordinates == False:
                height, width = frame.shape
                polygons_coordinates = setPolygonCoordinates(height, width)
                num_of_polygons = getNumOfPolygons(polygons_coordinates)
                calculate_coordinates = True

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if capture_images == True:
                if packet.stream_name == 'left':
                    filename = "left_" + str(total_num_of_captured_images) + ".png"
                    cv2.imwrite("dataset/left/" + str(filename), frame) 
                    print("py: Saved image as: " + str(filename))
                    captured_left_image = True

                elif packet.stream_name == 'right':
                    filename = "right_" + str(total_num_of_captured_images) + ".png"
                    cv2.imwrite("dataset/right/" + str(filename), frame) 
                    print("py: Saved image as: " + str(filename))
                    captured_right_image = True

                if captured_right_image == True and captured_left_image == True:
                    capture_images = False
                    captured_left_image = False
                    captured_right_image = False
                    total_num_of_captured_images += 1
                    image_per_polygon_counter += 1
                    
                    if image_per_polygon_counter == num_image_per_polygon:
                        polygon_index += 1
                        image_per_polygon_counter = 0

            if  num_of_polygons != polygon_index:                
                cv2.putText(frame, "Align cameras with callibration board and press spacebar to capture the image", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                cv2.putText(frame, "Number of saved images: " + str(total_num_of_captured_images), (750, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                cv2.polylines(frame, np.array([getPolygonCoordinates(polygon_index, polygons_coordinates)]), True, (0, 0, 255), 4)
                cv2.imshow(packet.stream_name, frame)
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
print("py: Starting calibration based on captured images")
cal_data = calibration_utils.StereoCalibration()
cal_data.calibrate("dataset", calib_puttern_size, "./depthai.calib")

print('py: DONE.')


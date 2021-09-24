import os
import subprocess
from time import sleep, time
from pathlib import Path
import sys
import serial
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

curr_path = Path(__file__).parent.resolve()
print(curr_path)
test_cmd = "esptool.py --chip esp32 --port /dev/ttyUSB0 --baud 921600 --before default_reset --after hard_reset write_flash -z --flash_mode dio --flash_freq 40m --flash_size detect 0x1000 " 
test_cmd += str(curr_path) + "/esp-test/bootloader.bin 0x10000 "
test_cmd += str(curr_path) + "/esp-test/esp-test-and-flash.bin 0x8000 "
test_cmd += str(curr_path) + "/esp-test/partition-table.bin"

p = subprocess.Popen(test_cmd, shell=True, preexec_fn=os.setsid)
# response_status.append(p)

# is_finished = False 
while p.poll() is None: pass

port = '/dev/ttyUSB0'
s = serial.Serial(port, 115200, timeout=5)

is_passed = False
start = time()
while s.is_open:
    str_val = s.read_until().decode("unicode_escape")
    end_str = '' if str_val[-1] == '\n' else '\n'
    print('[ESP32]', str_val, end=end_str)
    if 'TEST PASSED AND BO HEADER FLASHED!' in str_val:
        print(start)
        print(time())
        print(time() - start)
        is_passed = True
        break
    if (time() - start) > 10:
        is_passed = False
        break
    
s.close()

test_cmd = "esptool.py erase_region 0x0 0x40000"

p = subprocess.Popen(test_cmd, shell=True, preexec_fn=os.setsid)
# response_status.append(p)
# is_finished = False 
while p.poll() is None: pass

red = (255, 0, 0)
green = (0, 255, 0)
image = None

if is_passed:
    image = create_blank(512, 512, rgb_color=green)
    cv2.putText(image,'FLASH TEST',(10,250), font, 2,(0,0,0),2)
    cv2.putText(image,'PASSED',(10,300), font, 2,(0,0,0),2)
    cv2.putText(image,'GPIO test skipped!',(10,380), font, 1.5,(0,0,0),2)

else:
    image = create_blank(512, 512, rgb_color=red)
    cv2.putText(image,'FLASH TEST ',(10,250), font, 2,(0,0,0),2)
    cv2.putText(image,'FAILED',(10,300), font, 2,(0,0,0),2)

    

cv2.imshow("Result Image",image)
# Allow Ctrl-C to work
while cv2.waitKey(10) < 0: pass

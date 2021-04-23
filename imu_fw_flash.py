#!/usr/bin/env python3


import subprocess
import time
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
red = (255, 0, 0)
green = (0, 255, 0)
image = None

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

proc = subprocess.Popen(['python3','imu_main.py'],stdout=subprocess.PIPE)
isFlashed = False

start = time.time()
poll = None
while poll is None:
  line = proc.stdout.readline()
  if not line:
    break
  #the real code does filtering here
  logLine = line.rstrip().decode("unicode_escape")
  print("text log ->", logLine)
  if 'Part 10004148 : Version 3.9.7 Build 224' in logLine:
      isFlashed = True
      end = time.time()
      print(end - start)

if isFlashed:
    image = create_blank(512, 512, rgb_color=green)
    cv2.putText(image,'IMU FLASH',(10,250), font, 2,(0,0,0),2)
    cv2.putText(image,'PASSED',(10,300), font, 2,(0,0,0),2)
else:
    image = create_blank(512, 512, rgb_color=red)
    cv2.putText(image,'IMU FLASH ',(10,250), font, 2,(0,0,0),2)
    cv2.putText(image,'FAILED',(10,300), font, 2,(0,0,0),2)

cv2.imshow("Flash Result",image)
cv2.waitKey(0)
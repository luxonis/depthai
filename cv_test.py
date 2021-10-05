
import cv2
import numpy as np

red = (255, 0, 0)
green = (0, 255, 0)
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

image = create_blank(512, 512, rgb_color=green)
cv2.imshow("Result Image",image)
cv2.waitKey(0) 
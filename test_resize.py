import cv2
import numpy as np
import glob

images_left = glob.glob("dataset720" + '/left/*.png')
images_right = glob.glob("dataset720" + '/right/*.png')

for image_left, image_right in zip(images_left, images_right):
    # read images
    img_l = cv2.imread(image_left, 0)
    img_r = cv2.imread(image_right, 0)
    img_l = img_l[40:760, :]
    img_r = img_r[40:760, :]
    
    cv2.imwrite(image_left, img_l)
    cv2.imwrite(image_right, img_r)



images_left = glob.glob("dataset400" + '/left/*.png')
images_right = glob.glob("dataset400" + '/right/*.png')

for image_left, image_right in zip(images_left, images_right):
    # read images
    img_l = cv2.imread(image_left, 0)
    img_r = cv2.imread(image_right, 0)
    img_l = cv2.resize(img_l, (640, 400),interpolation = cv2.INTER_AREA)
    img_r = cv2.resize(img_r, (640, 400),interpolation = cv2.INTER_AREA)

    cv2.imwrite(image_left, img_l)
    cv2.imwrite(image_right, img_r)
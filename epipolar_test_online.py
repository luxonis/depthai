import math
import numpy as np
import cv2
from pathlib import Path
import depthai as dai


def filter_matches(kp_left, kp_right, des_left, des_right, matches, ratio = 0.75, reprojection_threshold = 5.0):
    # store all the good matches as per Lowe's ratio test.
    good = []
    pts_left_filtered = []
    pts_right_filtered = []
    kp_left_filtered = []
    kp_right_filtered = []
    des_left_filtered =  []
    des_right_filtered = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            pts_left_filtered.append(kp_left[m.queryIdx].pt)
            kp_left_filtered.append(kp_left[m.queryIdx])
            des_left_filtered.append(des_left[m.queryIdx])

            pts_right_filtered.append(kp_right[m.trainIdx].pt)
            kp_right_filtered.append(kp_right[m.trainIdx])
            des_right_filtered.append(des_right[m.trainIdx])

    if len(kp_left_filtered) < 25 or len(kp_right_filtered) < 25:
        return kp_left_filtered, kp_right_filtered, np.array(des_left_filtered), np.array(des_right_filtered)

    pts_left_filtered = np.float32(pts_left_filtered)
    pts_right_filtered = np.float32(pts_right_filtered)


    # this is just to get inliers
    M, mask = cv2.findHomography(pts_left_filtered, pts_right_filtered, method=cv2.RANSAC, ransacReprojThreshold=reprojection_threshold)
    matchesMask = mask.ravel().tolist()
    for i in reversed(range(len(pts_left_filtered))):
        if not matchesMask[i]:
            del kp_left_filtered[i]
            del kp_right_filtered[i]
            del des_left_filtered[i]
            del des_right_filtered[i]
    return kp_left_filtered, kp_right_filtered, np.array(des_left_filtered), np.array(des_right_filtered)

sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def detect_features(left_image, right_image):
    kp_left, des_left = sift.detectAndCompute(left_image, None)
    kp_right, des_right = sift.detectAndCompute(right_image, None)

    if len(kp_left) < 25 or len(kp_right) < 25:
        return None, None, None, None

    print(f'length of keypoints: {len(kp_left)}, {len(kp_right)}')
    matches = flann.knnMatch(des_left, des_right, k=2)

    filter_val = 0.6
    reprojection_threshold = 3.0
    kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered = filter_matches( kp_left, 
                    kp_right, 
                    des_left, 
                    des_right, 
                    matches, ratio = filter_val, reprojection_threshold=reprojection_threshold)

    print(f'length of filtered keypoints: {len(kp_left_filtered)}, {len(kp_right_filtered)}')

    if len(kp_left_filtered) < 25 or len(kp_right_filtered) < 25:
        return None, None, None, None
    return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered

def epipolar_calculate(kp_left_filtered, kp_right_filtered, left_undistorted, right_undistorted, size):

    horStack = np.hstack((left_undistorted, right_undistorted))
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    radius = 2
    thickness = 1
    epiploar_error = 0

    for i in range(len(kp_left_filtered)):
        left_pt = kp_left_filtered[i].pt
        right_pt = kp_right_filtered[i].pt
        
        left_pt_i = (int(left_pt[0]), int(left_pt[1]))
        right_pt_i = (size[0] + int(right_pt[0]), int(right_pt[1]))

        cv2.circle(horStack, left_pt_i, radius, red, thickness)
        cv2.circle(horStack, right_pt_i, radius, red, thickness)
        horStack = cv2.line(horStack, left_pt_i, right_pt_i, green, thickness)
        epiploar_error += abs(left_pt[1] - right_pt[1])
    
    epiploar_error /= len(kp_left_filtered)
    dest = cv2.resize(horStack, (0, 0), fx = 0.5, fy= 0.5, interpolation=cv2.INTER_AREA)
    return epiploar_error, dest

def getDevice(calib):

    device = dai.Device()
    if not calib:
        calibHandler = device.readCalibration()

    pipeline = dai.Pipeline()

    cams = device.getConnectedCameras()
    sensorNames = device.getCameraSensorNames()

    if not dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams:
        raise RuntimeError("Left and right cameras are not available for epipolar check")

    for cam in cams:
        if cam == dai.CameraBoardSocket.LEFT:
            name = sensorNames[dai.CameraBoardSocket.LEFT]
            camLeft = None
            print('Name of left camera: ', name)
            if name == 'OV9282':
                camLeft = pipeline.create(dai.node.MonoCamera)
                camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            xoutLeft = pipeline.create(dai.node.XLinkOut)
            xoutLeft.setStreamName("left")
            camLeft.out.link(xoutLeft.input)
        elif cam == dai.CameraBoardSocket.RIGHT:
            name = sensorNames[dai.CameraBoardSocket.RIGHT]
            camRight = None
            if name == 'OV9282':
                camRight = pipeline.create(dai.node.MonoCamera)
                camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            xoutRight = pipeline.create(dai.node.XLinkOut)
            xoutRight.setStreamName("right")
            camRight.out.link(xoutRight.input)

    device.startPipeline(pipeline)
    return device, calibHandler

def evaluateDevice(device, calibHandler):
    left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    left_k, w, h = calibHandler.getDefaultIntrinsics(dai.CameraBoardSocket.LEFT)
    right_k, _, _ = calibHandler.getDefaultIntrinsics(dai.CameraBoardSocket.RIGHT)
    left_k = np.array(left_k)
    right_k = np.array(right_k)
    left_d = np.array(calibHandler.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    right_d = np.array(calibHandler.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    left_r = np.array(calibHandler.getStereoLeftRectificationRotation())
    right_r = np.array(calibHandler.getStereoRightRectificationRotation())
    left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_k, left_d, left_r, right_k, (w, h), cv2.CV_16SC2)
    right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_k, right_d, right_r, right_k, (w, h), cv2.CV_16SC2)
    width = w
    height = h

    left_image = None
    right_image = None

    hor_epipolar_list = []
    while not device.isClosed():
        left_image = left_queue.get().getCvFrame()
        right_image = right_queue.get().getCvFrame()

        left_hor_undistorted = cv2.remap(left_image, left_mapx, left_mapy, cv2.INTER_LINEAR)
        right_hor_undistorted = cv2.remap(right_image, right_mapx, right_mapy, cv2.INTER_LINEAR)

        # cv2.imshow("left_hor_undistorted", left_hor_undistorted)
        # cv2.imshow("right_hor_undistorted", right_hor_undistorted)        
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        kp_left, kp_right, _, _ = detect_features(left_hor_undistorted, right_hor_undistorted)
        if kp_left is None or kp_right is None:
            print('Getting keypoints failed for horizontal stereo')
            continue

        hor_epipolar_error, hor_stack = epipolar_calculate(kp_left, kp_right, left_hor_undistorted, right_hor_undistorted, (width, height))
        hor_epipolar_list.append(hor_epipolar_error)
        print(f' average hor epipolar error per frame: {hor_epipolar_error}')

        cv2.imshow("hor_stack", hor_stack)
    print(f'Average hor Epiploar error across {len(hor_epipolar_list)} frames is : { sum(hor_epipolar_list) / len(hor_epipolar_list)}')



if __name__ == '__main__':
    calib = None
    device, calib = getDevice(calib)
    evaluateDevice(device, calib)



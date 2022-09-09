#!/usr/bin/env python3

description=\
"""
Dynamic recalibration script.
Capable of correcting extrinsic rotation (e.g. rotation change between sensors) without the need of full recalibration.
Recommended way of doing dynamic calibration is pointing the camera to a static scene, and running the script.
Recommended to try dynamic calibration if depth quality degraded over time.

Requires initial intrinsic calibration.
This script supports all sensor combinations that calibrate.py supports.
"""

from cmath import inf
import numpy as np
import cv2
import depthai as dai
import math
import argparse
from pathlib import Path

ransacMethod = cv2.RANSAC
if cv2.__version__ >= "4.5.4":
    ransacMethod = cv2.USAC_MAGSAC

epilog_text="Dynamic recalibration."
parser = argparse.ArgumentParser(
    epilog=epilog_text, description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-rd", "--rectifiedDisp", default=True, action="store_false",
                    help="Display rectified images with lines drawn for epipolar check")
parser.add_argument("-drgb", "--disableRgb", default=False, action="store_true",
                    help="Disable rgb camera Calibration")
parser.add_argument("-ep", "--maxEpiploarError", default="1.0", type=float, required=False,
                    help="Sets the maximum epiploar allowed with rectification")
parser.add_argument("-rlp", "--rgbLensPosition", default=None, type=int,
                    required=False, help="Set the manual lens position of the camera for calibration")
parser.add_argument("-fps", "--fps", default=10, type=int,
                    required=False, help="Set capture FPS for all cameras. Default: %(default)s")
parser.add_argument("-d", "--debug", default=False, action="store_true", help="Enable debug logs.")
parser.add_argument("-dr", "--dryRun", default=False, action="store_true", help="Dry run, don't flash obtained calib data, just save to disk.")
options = parser.parse_args()

#TODO implement RGB-stereo sync

epipolar_threshold = options.maxEpiploarError
rgbEnabled = not options.disableRgb
dryRun = options.dryRun
debug = options.debug

def calculate_Rt_from_frames(frame1,frame2,k1,k2,d1,d2):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1,None)
    kp2, des2 = sift.detectAndCompute(frame2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    minKeypoints = 20
    if len(pts1) < minKeypoints:
        raise Exception(f'Need at least {minKeypoints} keypoints!')

    if debug:
        img=cv2.drawKeypoints(frame1, kp1, frame1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Left", img)
        img2=cv2.drawKeypoints(frame2, kp2, frame2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Right", img2)
        cv2.waitKey(1)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    E, mask = cv2.findEssentialMat(pts1,pts2,k1,d1,k2,d2, method=ransacMethod)

    points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1,pts2, mask=mask)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(k1, d1, k2, d2, frame2.shape[::-1], R_est, t_est)

    return R_est, t_est, R1, R2, P1, P2, Q

def calculate_epipolar_error(frame1, frame2):
    minNrInliers = 10
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1,None)
    kp2, des2 = sift.detectAndCompute(frame2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if len(pts1) < minNrInliers or len(pts2) < minNrInliers:
        return math.inf

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    # this is just to get inliers
    M, mask = cv2.findHomography(pts1, pts2, method = ransacMethod, ransacReprojThreshold = 5.0)
    matchesMask = mask.ravel().tolist()

    epi_error_sum = 0
    for i in range(len(pts1)):
        if not matchesMask[i]:
            continue
        pt2 = pts2[i]
        pt1 = pts1[i]

        epi_error_sum += abs(pt1[1] - pt2[1])
    if len(pts1) < minNrInliers:
        return math.inf
    return epi_error_sum / len(pts1)

def display_rectification(image_data_pairs):
    print("Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
    for image_data_pair in image_data_pairs:
        pair0 = image_data_pair[0]
        pair1 = image_data_pair[1]
        if len(pair0.shape) < 3:
            pair0 = cv2.cvtColor(pair0, cv2.COLOR_GRAY2RGB)
        if len(pair1.shape) < 3:
            pair1 = cv2.cvtColor(pair1, cv2.COLOR_GRAY2RGB)
        img_concat = cv2.hconcat([pair0, pair1])

        # draw epipolar lines for debug purposes
        line_row = 0
        while line_row < img_concat.shape[0]:
            cv2.line(img_concat,
                        (0, line_row), (img_concat.shape[1], line_row),
                        (0, 255, 0), 1)
            line_row += 30

        # show image
        cv2.imshow('Stereo Pair', img_concat)
        k = cv2.waitKey(0)
        if k == 27:  # Esc key to stop
            break

    cv2.destroyWindow('Stereo Pair')

if __name__ == "__main__":

    camFps = options.fps

    pipeline = dai.Pipeline()
    device = dai.Device()

    try:
        calibration_handler = device.readCalibration2()
        original_calibration = device.readCalibration2()
    except Exception as e:
        print("Dynamic recalibration requires initial intrinsic calibration!")
        raise e


    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_right = pipeline.create(dai.node.MonoCamera)

    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_right = pipeline.create(dai.node.XLinkOut)

    xout_left_rect = pipeline.create(dai.node.XLinkOut)
    xout_right_rect = pipeline.create(dai.node.XLinkOut)
    stereo = pipeline.create(dai.node.StereoDepth)

    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    cam_left.setFps(camFps)

    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    cam_right.setFps(camFps)

    xout_left.setStreamName("left")
    xout_left_rect.setStreamName("left_rect")
    # cam_left.out.link(xout_left.input)

    xout_right.setStreamName("right")
    xout_right_rect.setStreamName("right_rect")
    # cam_right.out.link(xout_right.input)


    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)

    stereo.syncedLeft.link(xout_left.input)
    stereo.syncedRight.link(xout_right.input)
    stereo.rectifiedLeft.link(xout_left_rect.input)
    stereo.rectifiedRight.link(xout_right_rect.input)
    stereo_img_shape = cam_left.getResolutionSize()

    leftFps = cam_left.getFps()
    rightFps = cam_right.getFps()

    if leftFps != rightFps:
        raise Exception("FPS between left and right cameras must be the same!")

    if rgbEnabled:
        rgbLensPosition = None

        if options.rgbLensPosition:
            rgbLensPosition = options.rgbLensPosition
        else:
            try:
                rgbLensPosition = calibration_handler.getLensPosition(dai.CameraBoardSocket.RGB)
            except:
                pass

        rgb_cam = pipeline.createColorCamera()
        rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        rgb_cam.setInterleaved(False)
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_cam.setIspScale(1, 3)
        if rgbLensPosition:
            rgb_cam.initialControl.setManualFocus(rgbLensPosition)
        rgb_cam.setFps(camFps)

        xout_rgb_isp = pipeline.create(dai.node.XLinkOut)
        xout_rgb_isp.setStreamName("rgb")
        rgb_cam.isp.link(xout_rgb_isp.input)

        rgb_img_shape = rgb_cam.getVideoSize()

        rgbFps = rgb_cam.getFps()
        if leftFps != rgbFps:
            raise Exception("FPS between stereo cameras and rgb camera must be the same!")


    with device:
        device.startPipeline(pipeline)

        left_camera_queue = device.getOutputQueue("left", 4, True)
        right_camera_queue = device.getOutputQueue("right", 4, True)
        if rgbEnabled:
            rgb_camera_queue = device.getOutputQueue("rgb", 4, True)
        left_rectified_camera_queue = device.getOutputQueue("left_rect", 4, True)
        right_rectified_camera_queue = device.getOutputQueue("right_rect", 4, True)

        left_camera = dai.CameraBoardSocket.LEFT
        right_camera = dai.CameraBoardSocket.RIGHT
        rgb_camera = dai.CameraBoardSocket.RGB

        left_rect_frame = None
        right_rect_frame = None
        left_frame = None
        right_frame = None
        rgb_frame = None

        for i in range(2*int(leftFps)): #let the exposure settle
            left_rect_frame = left_rectified_camera_queue.get().getCvFrame()
            right_rect_frame = right_rectified_camera_queue.get().getCvFrame()

            leftFrameData = left_camera_queue.get()
            left_frame = leftFrameData.getCvFrame()
            rightFrameData = right_camera_queue.get()
            right_frame = rightFrameData.getCvFrame()

            stereo_img_shape = (leftFrameData.getWidth(), leftFrameData.getHeight())

            if rgbEnabled:
                rgbFrameData = rgb_camera_queue.get()
                rgb_frame = rgbFrameData.getCvFrame()
                rgb_img_shape = (rgbFrameData.getWidth(), rgbFrameData.getHeight())


        left_k = calibration_handler.getCameraIntrinsics(left_camera, stereo_img_shape[0], stereo_img_shape[1])
        right_k = calibration_handler.getCameraIntrinsics(right_camera, stereo_img_shape[0], stereo_img_shape[1])

        left_d = calibration_handler.getDistortionCoefficients(left_camera)
        right_d = calibration_handler.getDistortionCoefficients(right_camera)

        left_k = np.array(left_k)
        right_k = np.array(right_k)

        left_d = np.array(left_d)
        right_d = np.array(right_d)
        rotationLeft = np.array(calibration_handler.getStereoLeftRectificationRotation())
        rotationRight = np.array(calibration_handler.getStereoRightRectificationRotation())

        if rgbEnabled:
            rgb_k = calibration_handler.getCameraIntrinsics(rgb_camera, rgb_img_shape[0], rgb_img_shape[1])
            rgb_k = np.array(rgb_k)
            rgb_d = calibration_handler.getDistortionCoefficients(rgb_camera)
            rgb_d = np.array(rgb_d)

        while True:
            try:
                left_rect_frame = left_rectified_camera_queue.get().getCvFrame()
                right_rect_frame = right_rectified_camera_queue.get().getCvFrame()

                leftFrameData = left_camera_queue.get()
                left_frame = leftFrameData.getCvFrame()
                rightFrameData = right_camera_queue.get()
                right_frame = rightFrameData.getCvFrame()

                if rgbEnabled:
                    rgb_frame = rgb_camera_queue.get().getCvFrame()
                R, T, R1, R2, P1, P2, Q = calculate_Rt_from_frames(left_frame,right_frame,left_k,right_k,left_d,right_d)

                if rgbEnabled:
                    rgbR, rgbT, _, _, _, _, _ = calculate_Rt_from_frames(rgb_frame,right_frame,rgb_k,right_k,rgb_d,right_d)
                    rgbR = np.linalg.inv(rgbR) #right to rgb rotation

                img_shape = cam_left.getResolutionSize()
                M1 = left_k
                M2 = right_k
                d1 = left_d
                d2 = right_d

                mapx_l, mapy_l = cv2.initUndistortRectifyMap(M1, d1, R1, M2, img_shape, cv2.CV_32FC1)
                mapx_r, mapy_r = cv2.initUndistortRectifyMap(M2, d2, R2, M2, img_shape, cv2.CV_32FC1)

                img_l = cv2.remap(left_frame, mapx_l, mapy_l, cv2.INTER_LINEAR)
                img_r = cv2.remap(right_frame, mapx_r, mapy_r, cv2.INTER_LINEAR)


                stereo_epipolar = calculate_epipolar_error(img_l, img_r)
                if stereo_epipolar > epipolar_threshold:
                    print(f"Stereo epipolar error: {stereo_epipolar} is higher than threshold {epipolar_threshold}")
                    continue

                if rgbEnabled:
                    M3 = rgb_k
                    d3 = rgb_d
                    R3 = rgbR
                    mapx_rgb, mapy_rgb = cv2.initUndistortRectifyMap(M3, d3, None, M3, img_shape, cv2.CV_32FC1)
                    mapx_rgb2, mapy_rgb2 = cv2.initUndistortRectifyMap(M2, d2, R3, M3, img_shape, cv2.CV_32FC1)
                    img_rgb = cv2.remap(rgb_frame, mapx_rgb, mapy_rgb, cv2.INTER_LINEAR)
                    img_rgb2 = cv2.remap(right_frame, mapx_rgb2, mapy_rgb2, cv2.INTER_LINEAR)
                    rgb_epipolar = calculate_epipolar_error(img_rgb, img_rgb2)
                    if rgb_epipolar > epipolar_threshold:
                        print(f"RGB epipolar {rgb_epipolar} is higher than threshold {epipolar_threshold}")
                        continue

                break
            except Exception as e:
                print(e)
                continue

        print(f"Stereo epipolar error: {stereo_epipolar}")
        if rgbEnabled:
            print(f"RGB epipolar error: {rgb_epipolar}")


        #save rotation data
        lrSpecExtrinsics = calibration_handler.getCameraExtrinsics(left_camera, right_camera, True)
        specTranslation = (lrSpecExtrinsics[0][3], lrSpecExtrinsics[1][3], lrSpecExtrinsics[2][3])
        lrCompExtrinsics = calibration_handler.getCameraExtrinsics(left_camera, right_camera, False)
        compTranslation = (lrCompExtrinsics[0][3], lrCompExtrinsics[1][3], lrCompExtrinsics[2][3])
        calibration_handler.setCameraExtrinsics(left_camera, right_camera, R, compTranslation, specTranslation)

        calibration_handler.setStereoLeft(left_camera, R1)
        calibration_handler.setStereoRight(right_camera, R2)

        if rgbEnabled:
            rgbSpecExtrinsics = calibration_handler.getCameraExtrinsics(right_camera, rgb_camera, True)
            specTranslation = (rgbSpecExtrinsics[0][3], rgbSpecExtrinsics[1][3], rgbSpecExtrinsics[2][3])
            rgbCompExtrinsics = calibration_handler.getCameraExtrinsics(right_camera, rgb_camera, False)
            compTranslation = (rgbCompExtrinsics[0][3], rgbCompExtrinsics[1][3], rgbCompExtrinsics[2][3])
            calibration_handler.setCameraExtrinsics(right_camera, rgb_camera, rgbR, compTranslation, specTranslation)

        #flash updates

        is_write_successful = False
        if not dryRun:
            calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}_backup.json")).resolve().absolute())
            original_calibration.eepromToJsonFile(calibFile)
            print(f"Original calibration data on the device is backed up at: {calibFile}")

            is_write_successful = device.flashCalibration(calibration_handler)
            if not is_write_successful:
                print(f"Error: failed to save calibration to EEPROM")
        else:
            calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}_dynamic_calib.json")).resolve().absolute())
            calibration_handler.eepromToJsonFile(calibFile)
            print(f"Dynamic calibration data on the device is saved at: {calibFile}")


        if options.rectifiedDisp:
            image_data_pairs = []
            image_data_pairs.append((img_l, img_r))
            if rgbEnabled:
                image_data_pairs.append((img_rgb, img_rgb2))

            display_rectification(image_data_pairs)


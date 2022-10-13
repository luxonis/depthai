#!/usr/bin/env python3

import cv2
import glob
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import time
import json
import cv2.aruco as aruco
from pathlib import Path
# Creates a set of 13 polygon coordinates


def setPolygonCoordinates(height, width):
    horizontal_shift = width//4
    vertical_shift = height//4

    margin = 60
    slope = 150

    p_coordinates = [
        [[margin, margin], [margin, height-margin],
            [width-margin, height-margin], [width-margin, margin]],

        [[margin, 0], [margin, height], [width//2, height-slope], [width//2, slope]],
        [[horizontal_shift, 0], [horizontal_shift, height], [
            width//2 + horizontal_shift, height-slope], [width//2 + horizontal_shift, slope]],
        [[horizontal_shift*2-margin, 0], [horizontal_shift*2-margin, height], [width//2 +
                                                                               horizontal_shift*2-margin, height-slope], [width//2 + horizontal_shift*2-margin, slope]],

        [[width-margin, 0], [width-margin, height],
            [width//2, height-slope], [width//2, slope]],
        [[width-horizontal_shift, 0], [width-horizontal_shift, height], [width //
                                                                         2-horizontal_shift, height-slope], [width//2-horizontal_shift, slope]],
        [[width-horizontal_shift*2+margin, 0], [width-horizontal_shift*2+margin, height], [width //
                                                                                           2-horizontal_shift*2+margin, height-slope], [width//2-horizontal_shift*2+margin, slope]],

        [[0, margin], [width, margin], [
            width-slope, height//2], [slope, height//2]],
        [[0, vertical_shift], [width, vertical_shift], [width-slope,
                                                        height//2+vertical_shift], [slope, height//2+vertical_shift]],
        [[0, vertical_shift*2-margin], [width, vertical_shift*2-margin], [width-slope,
                                                                          height//2+vertical_shift*2-margin], [slope, height//2+vertical_shift*2-margin]],

        [[0, height-margin], [width, height-margin],
         [width-slope, height//2], [slope, height//2]],
        [[0, height-vertical_shift], [width, height-vertical_shift], [width -
                                                                      slope, height//2-vertical_shift], [slope, height//2-vertical_shift]],
        [[0, height-vertical_shift*2+margin], [width, height-vertical_shift*2+margin], [width -
                                                                                        slope, height//2-vertical_shift*2+margin], [slope, height//2-vertical_shift*2+margin]]
    ]
    return p_coordinates


def getPolygonCoordinates(idx, p_coordinates):
    return p_coordinates[idx]


def getNumOfPolygons(p_coordinates):
    return len(p_coordinates)

# Filters polygons to just those at the given indexes.


def select_polygon_coords(p_coordinates, indexes):
    if indexes == None:
        # The default
        return p_coordinates
    else:
        print("Filtering polygons to those at indexes=", indexes)
        return [p_coordinates[i] for i in indexes]


def image_filename(stream_name, polygon_index, total_num_of_captured_images):
    return "{stream_name}_p{polygon_index}_{total_num_of_captured_images}.png".format(stream_name=stream_name, polygon_index=polygon_index, total_num_of_captured_images=total_num_of_captured_images)


def polygon_from_image_name(image_name):
    """Returns the polygon index from an image name (ex: "left_p10_0.png" => 10)"""
    return int(re.findall("p(\d+)", image_name)[0])


class StereoCalibration(object):
    """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def __init__(self):
        """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def calibrate(self, board_config, filepath, square_size, mrk_size, squaresX, squaresY, camera_model, enable_disp_rectify):
        """Function to calculate calibration for stereo camera."""
        start_time = time.time()
        # init object data
        print(f'squareX is {squaresX}')
        self.enable_rectification_disp = enable_disp_rectify
        self.cameraModel = camera_model
        self.data_path = filepath
        self.aruco_dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        
        self.board = aruco.CharucoBoard_create(
            # 22, 16,
            squaresX, squaresY,
            square_size,
            mrk_size,
            self.aruco_dictionary)

        # parameters = aruco.DetectorParameters_create()
        assert mrk_size != None,  "ERROR: marker size not set"

        for camera in board_config['cameras'].keys():
            cam_info = board_config['cameras'][camera]
            print(
                '<------------Calibrating {} ------------>'.format(cam_info['name']))
            images_path = filepath + '/' + cam_info['name']
            ret, intrinsics, dist_coeff, _, _, size = self.calibrate_intrinsics(
                images_path, cam_info['hfov'])
            cam_info['intrinsics'] = intrinsics
            cam_info['dist_coeff'] = dist_coeff
            cam_info['size'] = size
            cam_info['reprojection_error'] = ret
            print(
                '<------------Camera Name: {} ------------>'.format(cam_info['name']))
            print("Reprojection error of {0}: {1}".format(cam_info['name'], ret))
            print("intrinsics of {0}: \n {1}".format(cam_info['name'], intrinsics))
            # print(intrinsics)
            # print(ret)

        for camera in board_config['cameras'].keys():
            left_cam_info = board_config['cameras'][camera]
            if 'extrinsics' in left_cam_info:
                if 'to_cam' in left_cam_info['extrinsics']:
                    left_cam = camera
                    right_cam = left_cam_info['extrinsics']['to_cam']
                    left_path = filepath + '/' + left_cam_info['name']

                    right_cam_info = board_config['cameras'][left_cam_info['extrinsics']['to_cam']]
                    right_path = filepath + '/' + right_cam_info['name']
                    print('<-------------Extrinsics calibration of {} and {} ------------>'.format(
                        left_cam_info['name'], right_cam_info['name']))

                    specTranslation = left_cam_info['extrinsics']['specTranslation']
                    rot = left_cam_info['extrinsics']['rotation']

                    translation = np.array(
                        [specTranslation['x'], specTranslation['y'], specTranslation['z']], dtype=np.float32)
                    rotation = R.from_euler(
                        'xyz', [rot['r'], rot['p'], rot['y']], degrees=True).as_matrix().astype(np.float32)

                    extrinsics = self.calibrate_extrinsics(left_path, right_path, left_cam_info['intrinsics'], left_cam_info[
                                                           'dist_coeff'], right_cam_info['intrinsics'], right_cam_info['dist_coeff'], translation, rotation)
                    if extrinsics[0] == -1:
                        return -1, extrinsics[1]

                    if board_config['stereo_config']['left_cam'] == left_cam and board_config['stereo_config']['right_cam'] == right_cam:
                        board_config['stereo_config']['rectification_left'] = extrinsics[3]
                        board_config['stereo_config']['rectification_right'] = extrinsics[4]
                    elif board_config['stereo_config']['left_cam'] == right_cam and board_config['stereo_config']['right_cam'] == left_cam:
                        board_config['stereo_config']['rectification_left'] = extrinsics[4]
                        board_config['stereo_config']['rectification_right'] = extrinsics[3]

                    """ for stereoObj in board_config['stereo_config']:

                        if stereoObj['left_cam'] == left_cam and stereoObj['right_cam'] == right_cam and stereoObj['main'] == 1:
                            stereoObj['rectification_left'] = extrinsics[3]
                            stereoObj['rectification_right'] = extrinsics[4] """

                    print('<-------------Epipolar error of {} and {} ------------>'.format(
                        left_cam_info['name'], right_cam_info['name']))
                    left_cam_info['extrinsics']['epipolar_error'] = self.test_epipolar_charuco(
                        left_path, right_path, left_cam_info['intrinsics'], left_cam_info['dist_coeff'], right_cam_info['intrinsics'], right_cam_info['dist_coeff'], extrinsics[2], extrinsics[3], extrinsics[4])


                    left_cam_info['extrinsics']['rotation_matrix'] = extrinsics[1]
                    left_cam_info['extrinsics']['translation'] = extrinsics[2]
                    left_cam_info['extrinsics']['stereo_error'] = extrinsics[0]

 
        return 1, board_config

    def analyze_charuco(self, images, scale_req=False, req_resolution=(800, 1280)):
        """
        Charuco base pose estimation.
        """
        # print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        all_marker_corners = []
        all_marker_ids = []
        all_recovered = []
        # decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        count = 0
        for im in images:
            # print("=> Processing image {0}".format(im))
            img_pth = Path(im)
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            expected_height = gray.shape[0]*(req_resolution[1]/gray.shape[1])

            if scale_req and not (gray.shape[0] == req_resolution[0] and gray.shape[1] == req_resolution[1]):
                if int(expected_height) == req_resolution[0]:
                    # resizing to have both stereo and rgb to have same
                    # resolution to capture extrinsics of the rgb-right camera
                    gray = cv2.resize(gray, req_resolution[::-1],
                                      interpolation=cv2.INTER_CUBIC)
                else:
                    # resizing and cropping to have both stereo and rgb to have same resolution
                    # to calculate extrinsics of the rgb-right camera
                    scale_width = req_resolution[1]/gray.shape[1]
                    dest_res = (
                        int(gray.shape[1] * scale_width), int(gray.shape[0] * scale_width))
                    gray = cv2.resize(
                        gray, dest_res, interpolation=cv2.INTER_CUBIC)
                    if gray.shape[0] < req_resolution[0]:
                        raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
                            gray.shape[0], req_resolution[0]))
                    # print(gray.shape[0] - req_resolution[0])
                    del_height = (gray.shape[0] - req_resolution[0]) // 2
                    # gray = gray[: req_resolution[0], :]
                    gray = gray[del_height: del_height + req_resolution[0], :]

                count += 1
            marker_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray, self.aruco_dictionary)
            marker_corners, ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(gray, self.board,
                                                                                    marker_corners, ids, rejectedCorners=rejectedImgPoints)
            print('{0} number of Markers corners detected in the image {1}'.format(
                len(marker_corners), img_pth.name))
            if len(marker_corners) > 0:
                res2 = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, ids, gray, self.board)

                # if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:

                    cv2.cornerSubPix(gray, res2[1],
                                     winSize=(5, 5),
                                     zeroZone=(-1, -1),
                                     criteria=criteria)
                    allCorners.append(res2[1])  # Charco chess corners
                    allIds.append(res2[2])  # charuco chess corner id's
                    all_marker_corners.append(marker_corners)
                    all_marker_ids.append(ids)
                    all_recovered.append(recoverd)
                else:
                    raise RuntimeError("Failed to detect markers in the image")
            else:
                print(im + " Not found")
                raise RuntimeError("Failed to detect markers in the image")

        imsize = gray.shape
        return allCorners, allIds, all_marker_corners, all_marker_ids, imsize, all_recovered

    def calibrate_intrinsics(self, image_files, hfov):
        image_files = glob.glob(image_files + "/*")
        image_files.sort()
        assert len(
            image_files) != 0, "ERROR: Images not read correctly, check directory"

        allCorners, allIds, _, _, imsize, _ = self.analyze_charuco(image_files)
        if self.cameraModel == 'perspective':
            ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = self.calibrate_camera_charuco(
                allCorners, allIds, imsize[::-1], hfov)
            # (Height, width)
            return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, imsize[::-1]
        else:
            print('Fisheye--------------------------------------------------')
            ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = self.calibrate_fisheye(
                allCorners, allIds, imsize[::-1])
            # (Height, width)
            return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, imsize[::-1]

    def calibrate_extrinsics(self, images_left, images_right, M_l, d_l, M_r, d_r, guess_translation, guess_rotation):
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        images_left = glob.glob(images_left + "/*")
        images_right = glob.glob(images_right + "/*")

        images_left.sort()
        images_right.sort()

        assert len(
            images_left) != 0, "ERROR: Images not found, check directory"
        assert len(
            images_right) != 0, "ERROR: Images not found, check directory"
        # print('Images from left and right')
        # print(images_left[0])
        # print(images_right[0])

        scale = None
        scale_req = False
        frame_left_shape = cv2.imread(images_left[0], 0).shape
        frame_right_shape = cv2.imread(images_right[0], 0).shape
        scalable_res = frame_left_shape
        scaled_res = frame_right_shape

        if frame_right_shape[0] < frame_left_shape[0] and frame_right_shape[1] < frame_left_shape[1]:
            scale_req = True
            scale = frame_right_shape[1] / frame_left_shape[1]
        elif frame_right_shape[0] > frame_left_shape[0] and frame_right_shape[1] > frame_left_shape[1]:
            scale_req = True
            scale = frame_left_shape[1] / frame_right_shape[1]
            scalable_res = frame_right_shape
            scaled_res = frame_left_shape

        if scale_req:
            scaled_height = scale * scalable_res[0]
            diff = scaled_height - scaled_res[0]
            # if scaled_height <  smaller_res[0]:
            if diff < 0:
                scaled_res = (int(scaled_height), scaled_res[1])

        print(f'Is scale Req: {scale_req}\n scale value: {scale} \n scalable Res: {scalable_res} \n scale Res: {scaled_res}')
        print("Original res Left :{}".format(frame_left_shape))
        print("Original res Right :{}".format(frame_right_shape))
        print("Scale res :{}".format(scaled_res))

        # scaled_res = (scaled_height, )
        M_lp = self.scale_intrinsics(M_l, frame_left_shape, scaled_res)
        M_rp = self.scale_intrinsics(M_r, frame_right_shape, scaled_res)

        # print("~~~~~~~~~~~ POSE ESTIMATION LEFT CAMERA ~~~~~~~~~~~~~")
        allCorners_l, allIds_l, _, _, imsize_l, _ = self.analyze_charuco(
            images_left, scale_req, scaled_res)

        # print("~~~~~~~~~~~ POSE ESTIMATION RIGHT CAMERA ~~~~~~~~~~~~~")
        allCorners_r, allIds_r, _, _, imsize_r, _ = self.analyze_charuco(
            images_right, scale_req, scaled_res)

        print(f'Image size of right side :{imsize_r}')
        print(f'Image size of left side :{imsize_l}')

        assert imsize_r == imsize_l, "Left and right resolution scaling is wrong"

        return self.calibrate_stereo(
            allCorners_l, allIds_l, allCorners_r, allIds_r, imsize_r, M_lp, d_l, M_rp, d_r, guess_translation, guess_rotation)

    def scale_intrinsics(self, intrinsics, originalShape, destShape):
        scale = destShape[1] / originalShape[1]
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        scaled_intrinsics = np.matmul(scale_mat, intrinsics)
        """ print("Scaled height offset : {}".format(
            (originalShape[0] * scale - destShape[0]) / 2))
        print("Scaled width offset : {}".format(
            (originalShape[1] * scale - destShape[1]) / 2)) """
        scaled_intrinsics[1][2] -= (originalShape[0]
                                    * scale - destShape[0]) / 2
        scaled_intrinsics[0][2] -= (originalShape[1]
                                    * scale - destShape[1]) / 2
        print('original_intrinsics')
        print(intrinsics)
        print('scaled_intrinsics')
        print(scaled_intrinsics)

        return scaled_intrinsics

    def fisheye_undistort_visualizaation(self, img_list, K, D, img_size):
        for im in img_list:
            # print(im)
            img = cv2.imread(im)
            # h, w = img.shape[:2]
            if self.cameraModel == 'perspective':
                map1, map2 = cv2.initUndistortRectifyMap(
                    K, D, np.eye(3), K, img_size, cv2.CV_32FC1)
            else:
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), K, img_size, cv2.CV_32FC1)

            undistorted_img = cv2.remap(
                img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imshow("undistorted", undistorted_img)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def calibrate_camera_charuco(self, allCorners, allIds, imsize, hfov):
        """
        Calibrates the camera using the dected corners.
        """
        f = imsize[0] / (2 * np.tan(np.deg2rad(hfov/2)))
        # TODO(sachin): Change the initialization to be initialized using the guess from fov
        print("CAMERA CALIBRATION")
        print(imsize)
        cameraMatrixInit = np.array([[f,    0.0,      imsize[0]/2],
                                     [0.0,     f,      imsize[1]/2],
                                     [0.0,   0.0,        1.0]])

        # cameraMatrixInit = np.array([[857.1668,    0.0,      643.9126],
        #                                  [0.0,     856.0823,  387.56018],
        #                                  [0.0,        0.0,        1.0]])
        """ if imsize[1] < 700:
            cameraMatrixInit = np.array([[400.0,    0.0,      imsize[0]/2],
                                         [0.0,     400.0,  imsize[1]/2],
                                         [0.0,        0.0,        1.0]])
        elif imsize[1] < 1100:
            cameraMatrixInit = np.array([[857.1668,    0.0,      643.9126],
                                         [0.0,     856.0823,  387.56018],
                                         [0.0,        0.0,        1.0]])
        else:
            cameraMatrixInit = np.array([[3819.8801,    0.0,     1912.8375],
                                         [0.0,     3819.8801, 1135.3433],
                                         [0.0,        0.0,        1.]]) """

        print("Camera Matrix initialization.............")
        print(cameraMatrixInit)

        distCoeffsInit = np.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
                 cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #     flags = (cv2.CALIB_RATIONAL_MODEL)
        (ret, camera_matrix, distortion_coefficients,
         rotation_vectors, translation_vectors,
         stdDeviationsIntrinsics, stdDeviationsExtrinsics,
         perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=self.board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 50000, 1e-9))
        print('Per View Errors...')
        print(perViewErrors)
        return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors

    def calibrate_fisheye(self, allCorners, allIds, imsize):
        one_pts = self.board.chessboardCorners
        obj_points = []
        for i in range(len(allIds)):
            obj_pts_sub = []
            for j in range(len(allIds[i])):
                obj_pts_sub.append(one_pts[allIds[i][j]])
            obj_points.append(np.array(obj_pts_sub, dtype=np.float32))

        cameraMatrixInit = np.array([[500,    0.0,      643.9126],
                                    [0.0,     500,  387.56018],
                                    [0.0,        0.0,        1.0]])

        print("Camera Matrix initialization.............")
        print(cameraMatrixInit)
        flags = 0
        flags = cv2.fisheye.CALIB_CHECK_COND
        distCoeffsInit = np.zeros((4, 1))
        term_criteria = (cv2.TERM_CRITERIA_COUNT +
                         cv2.TERM_CRITERIA_EPS, 10000, 1e-5)

        return cv2.fisheye.calibrate(obj_points, allCorners, imsize, cameraMatrixInit, distCoeffsInit, flags=flags, criteria=term_criteria)

    def calibrate_stereo(self, allCorners_l, allIds_l, allCorners_r, allIds_r, imsize, cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, t_in, r_in):
        left_corners_sampled = []
        right_corners_sampled = []
        obj_pts = []
        one_pts = self.board.chessboardCorners
        # print('allIds_l')
        # print(len(allIds_l))
        # print('allIds_r')
        # print(len(allIds_r))
        # print('allIds_l')
        # print(allIds_l)
        # print(allIds_r)

        for i in range(len(allIds_l)):
            left_sub_corners = []
            right_sub_corners = []
            obj_pts_sub = []
        #     if len(allIds_l[i]) < 70 or len(allIds_r[i]) < 70:
        #         continue
            for j in range(len(allIds_l[i])):
                idx = np.where(allIds_r[i] == allIds_l[i][j])
                if idx[0].size == 0:
                    continue
                left_sub_corners.append(allCorners_l[i][j])
                right_sub_corners.append(allCorners_r[i][idx])
                obj_pts_sub.append(one_pts[allIds_l[i][j]])
            if len(left_sub_corners) > 3 and len(right_sub_corners) > 3:
                obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
                left_corners_sampled.append(
                    np.array(left_sub_corners, dtype=np.float32))
                right_corners_sampled.append(
                    np.array(right_sub_corners, dtype=np.float32))
            else:
                return -1, "Stereo Calib failed due to less common features"

        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 50000, 1e-9)

        if self.cameraModel == 'perspective':
            flags = 0
            # flags |= cv2.CALIB_USE_EXTRINSIC_GUESS
            # print(flags)

            flags |= cv2.CALIB_FIX_INTRINSIC
            # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            flags |= cv2.CALIB_RATIONAL_MODEL
            # print(flags)
            print('Printing Extrinsics guesses...')
            print(r_in)
            print(t_in)

            ret, M1, d1, M2, d2, R, T, E, F, _ = cv2.stereoCalibrateExtended(
                obj_pts, left_corners_sampled, right_corners_sampled,
                cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, imsize,
                R=r_in, T=t_in, criteria=stereocalib_criteria, flags=flags)
            print('Printing Extrinsics res...')
            print(R)
            print(T)
            # ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            #     obj_pts, left_corners_sampled, right_corners_sampled,
            #     cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, imsize,
            #     criteria=stereocalib_criteria, flags=flags)
            # if np.absolute(T[1])  > 0.2:
                
            R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                cameraMatrix_l,
                distCoeff_l,
                cameraMatrix_r,
                distCoeff_r,
                imsize, R, T)
            # self.P_l = P_l
            # self.P_r = P_r
            
            return [ret, R, T, R_l, R_r]

        elif self.cameraModel == 'fisheye':
            # print(len(obj_pts))
            # print('obj_pts')
            # print(obj_pts)
            # print(len(left_corners_sampled))
            # print('left_corners_sampled')
            # print(left_corners_sampled)
            # print(len(right_corners_sampled))
            # print('right_corners_sampled')
            # print(right_corners_sampled)
            # for i in range(len(obj_pts)):
            #     print('---------------------')
            #     print(i)
            #     print(len(obj_pts[i]))
            #     print(len(left_corners_sampled[i]))
            #     print(len(right_corners_sampled[i]))
            flags = 0
            # flags |= cv2.CALIB_FIX_INTRINSIC
            # flags |= cv2.CALIB_RATIONAL_MODEL
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS # TODO(sACHIN): Try without intrinsic guess
            flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC # TODO(sACHIN): Try without intrinsic guess

            print('Fisyeye stereo model..................')
            print(len(cameraMatrix_l))
            print(len(cameraMatrix_r))
            print(len(distCoeff_l))
            print(len(distCoeff_r))
            
            ret, M1, d1, M2, d2, R, T, E, F = cv2.fisheye.stereoCalibrate(
                obj_pts, left_corners_sampled, right_corners_sampled,
                cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, imsize,
                flags=flags, criteria=stereocalib_criteria), None, None

            R_l, R_r, P_l, P_r, Q = cv2.fisheye.stereoRectify(
                cameraMatrix_l,
                distCoeff_l,
                cameraMatrix_r,
                distCoeff_r,
                imsize, R, T)

            return [ret, R, T, R_l, R_r]

    def display_rectification(self, image_data_pairs, isHorizontal):
        print(
            "Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
        for image_data_pair in image_data_pairs:
            if isHorizontal:
                img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
            else:
                img_concat = cv2.vconcat([image_data_pair[0], image_data_pair[1]])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)

            # draw epipolar lines for debug purposes

            line_row = 0
            while isHorizontal and line_row < img_concat.shape[0]:
                cv2.line(img_concat,
                         (0, line_row), (img_concat.shape[1], line_row),
                         (0, 255, 0), 1)
                line_row += 30

            line_col = 0
            while not isHorizontal and line_col < img_concat.shape[1]:
                cv2.line(img_concat,
                         (line_col, 0), (line_col, img_concat.shape[0]),
                         (0, 255, 0), 1)
                line_col += 40


            img_concat = cv2.resize(
                    img_concat, (0, 0), fx=0.4, fy=0.4)
            
            # show image
            cv2.imshow('Stereo Pair', img_concat)
            k = cv2.waitKey(0)
            if k == 27:  # Esc key to stop
                break

                # os._exit(0)
                # raise SystemExit()

        cv2.destroyWindow('Stereo Pair')

    def scale_image(self, img, scaled_res):
        expected_height = img.shape[0]*(scaled_res[1]/img.shape[1])
        # print("Expected Height: {}".format(expected_height))

        if not (img.shape[0] == scaled_res[0] and img.shape[1] == scaled_res[1]):
            if int(expected_height) == scaled_res[0]:
                # resizing to have both stereo and rgb to have same
                # resolution to capture extrinsics of the rgb-right camera
                img = cv2.resize(img, (scaled_res[1], scaled_res[0]),
                                 interpolation=cv2.INTER_CUBIC)
                return img
            else:
                # resizing and cropping to have both stereo and rgb to have same resolution
                # to calculate extrinsics of the rgb-right camera
                scale_width = scaled_res[1]/img.shape[1]
                dest_res = (
                    int(img.shape[1] * scale_width), int(img.shape[0] * scale_width))
                img = cv2.resize(
                    img, dest_res, interpolation=cv2.INTER_CUBIC)
                if img.shape[0] < scaled_res[0]:
                    raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
                        img.shape[0], scaled_res[0]))
                # print(gray.shape[0] - req_resolution[0])
                del_height = (img.shape[0] - scaled_res[0]) // 2
                # gray = gray[: req_resolution[0], :]
                img = img[del_height: del_height + scaled_res[0], :]
                return img
        else:
            return img

    def test_epipolar_charuco(self, left_img_pth, right_img_pth, M_l, d_l, M_r, d_r, t, r_l, r_r):
        images_left = glob.glob(left_img_pth + '/*.png')
        images_right = glob.glob(right_img_pth + '/*.png')
        images_left.sort()
        images_right.sort()
        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"
        isHorizontal = True
        if np.absolute(t[1])  > 0.2:
            isHorizontal = False    
        
        scale = None
        scale_req = False
        frame_left_shape = cv2.imread(images_left[0], 0).shape
        frame_right_shape = cv2.imread(images_right[0], 0).shape
        scalable_res = frame_left_shape
        scaled_res = frame_right_shape
        if frame_right_shape[0] < frame_left_shape[0] and frame_right_shape[1] < frame_left_shape[1]:
            scale_req = True
            scale = frame_right_shape[1] / frame_left_shape[1]
        elif frame_right_shape[0] > frame_left_shape[0] and frame_right_shape[1] > frame_left_shape[1]:
            scale_req = True
            scale = frame_left_shape[1] / frame_right_shape[1]
            scalable_res = frame_right_shape
            scaled_res = frame_left_shape
        
        if scale_req:
            scaled_height = scale * scalable_res[0]
            diff = scaled_height - scaled_res[0]
            # if scaled_height <  smaller_res[0]:
            if diff < 0:
                scaled_res = (int(scaled_height), scaled_res[1])

        print(f'Is scale Req: {scale_req}\n scale value: {scale} \n scalable Res: {scalable_res} \n scale Res: {scaled_res}')
        print("Original res Left :{}".format(frame_left_shape))
        print("Original res Right :{}".format(frame_right_shape))
        # print("Scale res :{}".format(scaled_res))

        M_lp = self.scale_intrinsics(M_l, frame_left_shape, scaled_res)
        M_rp = self.scale_intrinsics(M_r, frame_right_shape, scaled_res)

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)

        # print('Scaled Res :{}'.format(scaled_res))
        mapx_l, mapy_l = cv2.initUndistortRectifyMap(
            M_lp, d_l, r_l, M_rp, scaled_res[::-1], cv2.CV_32FC1)
        mapx_r, mapy_r = cv2.initUndistortRectifyMap(
            M_rp, d_r, r_r, M_rp, scaled_res[::-1], cv2.CV_32FC1)

        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            img_l = self.scale_image(img_l, scaled_res)
            img_r = self.scale_image(img_r, scaled_res)
            # print(img_l.shape)
            # print(img_r.shape)

            # warp right image
            # img_l = cv2.warpPerspective(img_l, self.H1, img_l.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            # img_r = cv2.warpPerspective(img_r, self.H2, img_r.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            img_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)

            image_data_pairs.append((img_l, img_r))

        # compute metrics
        imgpoints_r = []
        imgpoints_l = []
        for i, image_data_pair in enumerate(image_data_pairs):
            #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners_l, ids_l, rejectedImgPoints = cv2.aruco.detectMarkers(
                image_data_pair[0], self.aruco_dictionary)
            marker_corners_l, ids_l, _, _ = cv2.aruco.refineDetectedMarkers(image_data_pair[0], self.board,
                                                                            marker_corners_l, ids_l,
                                                                            rejectedCorners=rejectedImgPoints)

            marker_corners_r, ids_r, rejectedImgPoints = cv2.aruco.detectMarkers(
                image_data_pair[1], self.aruco_dictionary)
            marker_corners_r, ids_r, _, _ = cv2.aruco.refineDetectedMarkers(image_data_pair[1], self.board,
                                                                            marker_corners_r, ids_r,
                                                                            rejectedCorners=rejectedImgPoints)

            res2_l = cv2.aruco.interpolateCornersCharuco(
                marker_corners_l, ids_l, image_data_pair[0], self.board)
            res2_r = cv2.aruco.interpolateCornersCharuco(
                marker_corners_r, ids_r, image_data_pair[1], self.board)

            img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)
            line_row = 0
            while line_row < img_concat.shape[0]:
                cv2.line(img_concat,
                         (0, line_row), (img_concat.shape[1], line_row),
                         (0, 255, 0), 1)
                line_row += 30

            # cv2.imshow('Stereo Pair', img_concat)
            # k = cv2.waitKey(0)
            # if k == 27:  # Esc key to stop
            #     break

            if res2_l[1] is not None and res2_r[2] is not None and len(res2_l[1]) > 3 and len(res2_r[1]) > 3:

                cv2.cornerSubPix(image_data_pair[0], res2_l[1],
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
                cv2.cornerSubPix(image_data_pair[1], res2_r[1],
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)

                # termination criteria
                img_pth = Path(images_right[i])
                corners_l = []
                corners_r = []
                for j in range(len(res2_l[2])):
                    idx = np.where(res2_r[2] == res2_l[2][j])
                    if idx[0].size == 0:
                        continue
                    corners_l.append(res2_l[1][j])
                    corners_r.append(res2_r[1][idx])

                imgpoints_l.extend(corners_l)
                imgpoints_r.extend(corners_r)
                epi_error_sum = 0
                for l_pt, r_pt in zip(corners_l, corners_r):
                    if isHorizontal:
                        epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])
                    else:
                        epi_error_sum += abs(l_pt[0][0] - r_pt[0][0])

                print("Average Epipolar Error per image on host in " + img_pth.name + " : " +
                      str(epi_error_sum / len(corners_l)))
            else:
                print('Numer of corners is in left -> {} and right -> {}'.format(len(marker_corners_l), len(marker_corners_r)))
                return -1

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            if isHorizontal:
                epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])
            else:
                epi_error_sum += abs(l_pt[0][0] - r_pt[0][0])

        avg_epipolar = epi_error_sum / len(imgpoints_r)
        print("Average Epipolar Error is : " + str(avg_epipolar))

        if self.enable_rectification_disp:
            self.display_rectification(image_data_pairs, isHorizontal)

        return avg_epipolar

    def create_save_mesh(self):  # , output_path):

        curr_path = Path(__file__).parent.resolve()
        print("Mesh path")
        print(curr_path)

        map_x_l, map_y_l = cv2.initUndistortRectifyMap(
            self.M1, self.d1, self.R1, self.M2, self.img_shape, cv2.CV_32FC1)
        map_x_r, map_y_r = cv2.initUndistortRectifyMap(
            self.M2, self.d2, self.R2, self.M2, self.img_shape, cv2.CV_32FC1)

        """ 
        map_x_l_fp32 = map_x_l.astype(np.float32)
        map_y_l_fp32 = map_y_l.astype(np.float32)
        map_x_r_fp32 = map_x_r.astype(np.float32)
        map_y_r_fp32 = map_y_r.astype(np.float32)
        
                
        print("shape of maps")
        print(map_x_l.shape)
        print(map_y_l.shape)
        print(map_x_r.shape)
        print(map_y_r.shape) """

        meshCellSize = 16
        mesh_left = []
        mesh_right = []

        for y in range(map_x_l.shape[0] + 1):
            if y % meshCellSize == 0:
                row_left = []
                row_right = []
                for x in range(map_x_l.shape[1] + 1):
                    if x % meshCellSize == 0:
                        if y == map_x_l.shape[0] and x == map_x_l.shape[1]:
                            row_left.append(map_y_l[y - 1, x - 1])
                            row_left.append(map_x_l[y - 1, x - 1])
                            row_right.append(map_y_r[y - 1, x - 1])
                            row_right.append(map_x_r[y - 1, x - 1])
                        elif y == map_x_l.shape[0]:
                            row_left.append(map_y_l[y - 1, x])
                            row_left.append(map_x_l[y - 1, x])
                            row_right.append(map_y_r[y - 1, x])
                            row_right.append(map_x_r[y - 1, x])
                        elif x == map_x_l.shape[1]:
                            row_left.append(map_y_l[y, x - 1])
                            row_left.append(map_x_l[y, x - 1])
                            row_right.append(map_y_r[y, x - 1])
                            row_right.append(map_x_r[y, x - 1])
                        else:
                            row_left.append(map_y_l[y, x])
                            row_left.append(map_x_l[y, x])
                            row_right.append(map_y_r[y, x])
                            row_right.append(map_x_r[y, x])
                if (map_x_l.shape[1] % meshCellSize) % 2 != 0:
                    row_left.append(0)
                    row_left.append(0)
                    row_right.append(0)
                    row_right.append(0)

                mesh_left.append(row_left)
                mesh_right.append(row_right)

        mesh_left = np.array(mesh_left)
        mesh_right = np.array(mesh_right)
        left_mesh_fpath = str(curr_path) + '/../resources/left_mesh.calib'
        right_mesh_fpath = str(curr_path) + '/../resources/right_mesh.calib'
        mesh_left.tofile(left_mesh_fpath)
        mesh_right.tofile(right_mesh_fpath)

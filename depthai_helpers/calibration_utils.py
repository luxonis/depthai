import cv2
import glob
import os
import shutil
import numpy as np
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
        [[margin, 0], [margin, height], [width//2, height-slope], [width//2, slope]],
        [[horizontal_shift, 0], [horizontal_shift, height], [
            width//2 + horizontal_shift, height-slope], [width//2 + horizontal_shift, slope]],
        [[horizontal_shift*2-margin, 0], [horizontal_shift*2-margin, height], [width//2 +
                                                                               horizontal_shift*2-margin, height-slope], [width//2 + horizontal_shift*2-margin, slope]],

        [[margin, margin], [margin, height-margin],
         [width-margin, height-margin], [width-margin, margin]],

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

    def calibrate(self, filepath, square_size, out_filepath, type, squaresX, squaresY, mrk_size=None):
        """Function to calculate calibration for stereo camera."""
        start_time = time.time()
        # init object data
        self.calibrate_rgb = True
        if type == 'charuco':
            self.aruco_dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
            # parameters = aruco.DetectorParameters_create()
            assert mrk_size != None,  "ERROR: marker size not set"
            self.board = aruco.CharucoBoard_create(
                # 22, 16,
                squaresX, squaresY,
                square_size,
                mrk_size,
                self.aruco_dictionary)
            # self.board = aruco.CharucoBoard_create(
            #     11, 8,
            #     square_size,
            #     mrk_size,
            #     self.aruco_dictionary)
            self.data_path = filepath
            self.calibrate_charuco3D(filepath)
        else:
            self.objp = np.zeros((9 * 6, 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
            for pt in self.objp:
                pt *= square_size

            # process images, detect corners, refine and save data
            self.process_images(filepath)
            self.calibrate_camera()
            # run calibration procedure and construct Homography and mesh
            # self.stereo_calibrate_two_homography_calib()
        print('~~~~~ Starting Stereo Calibratin ~~~~~')
        # self.stereo_calib_two_homo()

        # rgb-right extrinsic calibration
        if self.calibrate_rgb:
            # if True:
            # things to do.
            # First: change the center and other thigns of the calibration matrix of rgb camera
            self.rgb_calibrate(filepath)
        else:
            self.M3 = np.zeros((3, 3), dtype=np.float32)
            self.R_rgb = np.zeros((3, 3), dtype=np.float32)
            self.T_rgb = np.zeros(3, dtype=np.float32)
            self.d3 = np.zeros(14, dtype=np.float32)

        # self.M3_scaled_write = np.copy(self.M3_scaled)
        # self.M3_scaled_write[1, 2] += 40

        self.M3_nan = np.zeros((3, 3), dtype=np.float32)
        self.R_rgb_nan = np.zeros((3, 3), dtype=np.float32)
        self.T_rgb_nan = np.zeros(3, dtype=np.float32)
        self.d3_nan = np.zeros(14, dtype=np.float32)

        R1_fp32 = self.R1_rgb.astype(np.float32)
        R2_fp32 = self.R2_rgb.astype(np.float32)
        M1_fp32 = self.M1.astype(np.float32)
        M3_fp32 = self.M3.astype(np.float32)
        R_rgb_fp32 = self.R_rgb.astype(np.float32) 
        T_rgb_fp32 = self.T_rgb.astype(np.float32) 
        d1_coeff_fp32 = self.d1.astype(np.float32)
        d3_coeff_fp32 = self.d3.astype(np.float32)

        R_rgb_fp32 = np.linalg.inv(R_rgb_fp32) 
        T_rgb_fp32[0] = -T_rgb_fp32[0] 
        T_rgb_fp32[1] = -T_rgb_fp32[1]
        T_rgb_fp32[2] = -T_rgb_fp32[2]

        self.calib_data = [R1_fp32, R2_fp32, M1_fp32, M3_fp32,
                     R_rgb_fp32, T_rgb_fp32, d1_coeff_fp32, d3_coeff_fp32]
        
        if 1:  # Print matrices, to compare with device data
            np.set_printoptions(suppress=True, precision=6)
            print("\nR1 (left)")
            print(R1_fp32)
            print("\nR2 (rgb)")
            print(R2_fp32)
            print("\nM1 (left)")
            print(M1_fp32)
            print("\nM3 (rgb)")
            print(M3_fp32)
            print("\R (rgb)")
            print(R_rgb_fp32)
            print("\nT (rgb)")
            print(T_rgb_fp32)

        if 0:  # Print computed homography, to compare with device data
            np.set_printoptions(suppress=True, precision=6)
            for res_height in [800, 720, 400]:
                m1 = np.copy(M1_fp32)
                m2 = np.copy(M2_fp32)
                if res_height == 720:
                    m1[1, 2] -= 40
                    m2[1, 2] -= 40
                if res_height == 400:
                    m_scale = [[0.5,   0, 0],
                               [0, 0.5, 0],
                               [0,   0, 1]]
                    m1 = np.matmul(m_scale, m1)
                    m2 = np.matmul(m_scale, m2)
                h1 = np.matmul(np.matmul(m2, R1_fp32), np.linalg.inv(m1))
                h2 = np.matmul(np.matmul(m2, R2_fp32), np.linalg.inv(m2))
                h1 = np.linalg.inv(h1)
                h2 = np.linalg.inv(h2)
                print('\nHomography H1, H2 for height =', res_height)
                print(h1)
                print()
                print(h2)

        # self.create_save_mesh()

        # append specific flags to file
        # with open(out_filepath, "ab") as fp:
        #     fp.write(bytearray(flags))

        print("Calibration file written to %s." % (out_filepath))
        print("\tTook %i seconds to run image processing." %
              (round(time.time() - start_time, 2)))

        if type == 'charuco':
            if self.calibrate_rgb:
                avg_epipolar_rgb_r = self.test_epipolar_charuco_rgb(filepath)
            return avg_epipolar_rgb_r, self.calib_data


    def parse_frame(self, frame, stream_name, file_name):
        file_name += '.png'
        # filename = image_filename(stream_name, self.current_polygon, self.images_captured)
        # print(self.data_path + "/{}/{}".format(stream_name, file_name))
        ds_path = self.data_path + "/{}".format(stream_name)
        print(ds_path)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)

        cv2.imwrite(self.data_path +
                    "/{}/{}".format(stream_name, file_name), frame)
        print("py: Saved image as: " + str(file_name))
        return True

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
            print("=> Processing image {0}".format(im))
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.flip(gray, 0) # TODO(Sachin) : remove this later
            # width = scale[1]
            expected_height = gray.shape[0]*(req_resolution[1]/gray.shape[1])
            # print('expected height -------------------> ' + str(expected_height))
            # print('required height -------------------> ' +
            #       str(req_resolution))

            if scale_req and not (gray.shape[0] == req_resolution[0] and gray.shape[1] == req_resolution[1]):
                # print("scaling----------------------->")
                if int(expected_height) == req_resolution[0]:
                    # resizing to have both stereo and rgb to have same
                    # resolution to capture extrinsics of the rgb-right camera
                    gray = cv2.resize(gray, req_resolution[::-1],
                                      interpolation=cv2.INTER_CUBIC)
                    print('~~~~~~~~~~ Only resizing.... ~~~~~~~~~~~~~~~~')
                else:
                    # resizing and cropping to have both stereo and rgb to have same resolution
                    # to calculate extrinsics of the rgb-right camera
                    scale_width = req_resolution[1]/gray.shape[1]
                    dest_res = (
                        int(gray.shape[1] * scale_width), int(gray.shape[0] * scale_width))
                    # print("destination resolution------>")
                    # print(dest_res)
                    gray = cv2.resize(
                        gray, dest_res, interpolation=cv2.INTER_CUBIC)
                    # print("scaled gray shape")
                    # print(gray.shape)
                    if gray.shape[0] < req_resolution[0]:
                        raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
                            gray.shape[0], req_resolution[0]))
                    # print(gray.shape[0] - req_resolution[0])
                    del_height = (gray.shape[0] - req_resolution[0]) // 2
                    # gray = gray[: req_resolution[0], :]
                    gray = gray[del_height: del_height + req_resolution[0], :]

                    print("del height ??")
                    print(del_height)
                    print(gray.shape)
                count += 1
                self.parse_frame(gray, 'rgb_resized',
                                 'rgb_resized_'+str(count))
            marker_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray, self.aruco_dictionary)
            marker_corners, ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(gray, self.board,
                                                                                    marker_corners, ids, rejectedCorners=rejectedImgPoints)
            print('{0} number of Markers corners detected in the above image'.format(
                len(marker_corners)))
            if len(marker_corners) > 0:
                # print(len(marker_corners))
                # SUB PIXEL DETECTION
                #             for corner in marker_corners:
                #                 cv2.cornerSubPix(gray, corner,
                #                                  winSize = (5,5),
                #                                  zeroZone = (-1,-1),
                #                                  criteria = criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, ids, gray, self.board)

                # if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:

                    cv2.cornerSubPix(gray, res2[1],
                                     winSize=(5, 5),
                                     zeroZone=(-1, -1),
                                     criteria=criteria)
                    allCorners.append(res2[1])
                    allIds.append(res2[2])
                    all_marker_corners.append(marker_corners)
                    all_marker_ids.append(ids)
                    all_recovered.append(recoverd)
                else:
                    print("in else")
            else:
                print(im + " Not found")
            # decimator+=1

        imsize = gray.shape
        return allCorners, allIds, all_marker_corners, all_marker_ids, imsize, all_recovered

    def calibrate_charuco3D(self, filepath):
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        # calcorners_l = []  # 2d points in image
        # calcorners_r = []  # 2d points in image
        # calids_l = []  # ids found in imag
        # calids_r = []  # ids found in imag

        images_left = glob.glob(filepath + "/left/*")
        # images_right = glob.glob(filepath + "/right/*")
        images_rgb = glob.glob(filepath + "/rgb/*")
        # print("Images left path------------------->")
        # print(images_left)
        images_left.sort()
        # images_right.sort()
        images_rgb.sort()

        assert len(
            images_left) != 0, "ERROR: Images not read correctly, check directory"
        # assert len(
        #     images_right) != 0, "ERROR: Images not read correctly, check directory"
        assert len(
            images_rgb) != 0, "ERROR: Images not read correctly, check directory"

        print("~~~~~~~~~~~ POSE ESTIMATION LEFT CAMERA ~~~~~~~~~~~~~")
        allCorners_l, allIds_l, _, _, imsize, _ = self.analyze_charuco(
            images_left)
        
        self.img_shape = imsize[::-1]
        # self.img_shape_rgb = imsize_rgb[::-1]
        ret_l, self.M1, self.d1, rvecs, tvecs = self.calibrate_camera_charuco(
            allCorners_l, allIds_l, self.img_shape)

        print("~~~~~~~~~~~~~RMS error of left~~~~~~~~~~~~~~")
        print(ret_l)

        ''' 
        COmmenting out RGB calibration here............
        if self.calibrate_rgb:
            print("~~~~~~~~~~~ POSE ESTIMATION RGB CAMERA FULL RES~~~~~~~~~~~~~")
            allCorners_rgb, allIds_rgb, _, _, imsize_rgb, _ = self.analyze_charuco(
                images_rgb)

            ret_rgb, self.M3, self.d3, rvecs, tvecs = self.calibrate_camera_charuco(
                allCorners_rgb, allIds_rgb, imsize_rgb[::-1])

        print("~~~~~~~~~~~~~RMS error of RGB, right and left~~~~~~~~~~~~~~")
        if self.calibrate_rgb:
            print(ret_rgb)
        print(ret_l)
        '''
        
        # left_corners_sampled = []
        # right_corners_sampled = []
        # obj_pts = []
        # one_pts = self.board.chessboardCorners
        # for i in range(len(allIds_l)):
        #     left_sub_corners = []
        #     right_sub_corners = []
        #     obj_pts_sub = []
        # #     if len(allIds_l[i]) < 70 or len(allIds_r[i]) < 70:
        # #         continue
        #     for j in range(len(allIds_l[i])):
        #         idx = np.where(allIds_r[i] == allIds_l[i][j])
        #         if idx[0].size == 0:
        #             continue
        #         left_sub_corners.append(allCorners_l[i][j])
        #         right_sub_corners.append(allCorners_r[i][idx])
        #         obj_pts_sub.append(one_pts[allIds_l[i][j]])

        #     obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
        #     left_corners_sampled.append(
        #         np.array(left_sub_corners, dtype=np.float32))
        #     right_corners_sampled.append(
        #         np.array(right_sub_corners, dtype=np.float32))

        # self.objpoints = obj_pts
        # self.imgpoints_l = left_corners_sampled
        # self.imgpoints_r = right_corners_sampled


    def calibrate_camera_charuco(self, allCorners, allIds, imsize):
        """
        Calibrates the camera using the dected corners.
        """
        print("CAMERA CALIBRATION")
        print(imsize)
        if imsize[1] < 1100:
            cameraMatrixInit = np.array([[857.1668,    0.0,      643.9126],
                                         [0.0,     856.0823,  387.56018],
                                         [0.0,        0.0,        1.0]])
        else:
            cameraMatrixInit = np.array([[3819.8801,    0.0,     1912.8375],
                                         [0.0,     3819.8801, 1135.3433],
                                         [0.0,        0.0,        1.]])
        
        print("Camera Matrix initialization.............")
        print(cameraMatrixInit)

        distCoeffsInit = np.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
                 cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #     flags = (cv2.CALIB_RATIONAL_MODEL)
        (ret, camera_matrix, distortion_coefficients0,
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
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

    def rgb_calibrate(self, filepath):
        images_left = glob.glob(filepath + "/left/*")
        images_rgb = glob.glob(filepath + "/rgb/*")

        images_left.sort()
        images_rgb.sort()

        allCorners_rgb_scaled, allIds_rgb_scaled, _, _, imsize_rgb_scaled, _ = self.analyze_charuco(
            images_rgb, scale_req=True, req_resolution=(720, 1280))
        self.img_shape_rgb_scaled = imsize_rgb_scaled[::-1]

        ret_rgb_scaled, self.M3_scaled, self.d3_scaled, rvecs, tvecs = self.calibrate_camera_charuco(
            allCorners_rgb_scaled, allIds_rgb_scaled, imsize_rgb_scaled[::-1])

        allCorners_r_rgb, allIds_r_rgb, _, _, _, _ = self.analyze_charuco(
            images_left, scale_req=True, req_resolution=(720, 1280))

        print("RGB callleded RMS at 720")
        # print(ret_rgb_scaled)
        print(imsize_rgb_scaled)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print(self.M3_scaled)

        # sampling common detected corners
        rgb_scaled_rgb_corners_sampled = []
        rgb_scaled_right_corners_sampled = []
        rgb_scaled_obj_pts = []
        rgb_pts = None
        right_pts = None
        one_pts = self.board.chessboardCorners
        for i in range(len(allIds_rgb_scaled)):
            rgb_sub_corners = []
            right_sub_corners = []
            obj_pts_sub = []
            # if len(allIds_l[i]) < 70 or len(allIds_r[i]) < 70:
            #     continue
            for j in range(len(allIds_rgb_scaled[i])):
                idx = np.where(allIds_r_rgb[i] == allIds_rgb_scaled[i][j])
                if idx[0].size == 0:
                    continue
                rgb_sub_corners.append(allCorners_rgb_scaled[i][j])
                right_sub_corners.append(allCorners_r_rgb[i][idx])
                obj_pts_sub.append(one_pts[allIds_rgb_scaled[i][j]])

            rgb_scaled_obj_pts.append(
                np.array(obj_pts_sub, dtype=np.float32))
            rgb_scaled_rgb_corners_sampled.append(
                np.array(rgb_sub_corners, dtype=np.float32))
            rgb_scaled_right_corners_sampled.append(
                np.array(right_sub_corners, dtype=np.float32))
            if rgb_pts is None:
                rgb_pts = np.array(rgb_sub_corners, dtype=np.float32)
                right_pts = np.array(right_sub_corners, dtype=np.float32)
            else:
                np.vstack(
                    (rgb_pts, np.array(rgb_sub_corners, dtype=np.float32)))
                np.vstack((right_pts, np.array(
                    right_sub_corners, dtype=np.float32)))

        self.objpoints_rgb_r = rgb_scaled_obj_pts
        self.imgpoints_rgb = rgb_scaled_rgb_corners_sampled
        self.imgpoints_rgb_right = rgb_scaled_right_corners_sampled
        
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_RATIONAL_MODEL


        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                    cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # print(M_RGB)
        print('vs. intrinisics computed after scaling the image --->')
        # self.M3, self.d3
        scale = 1920/1280
        print(scale)
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        self.M3 = np.matmul(scale_mat, self.M3_scaled)
        self.d3 = self.d3_scaled
        print(self.M3_scaled)
        print(self.M3)

        self.M2_rgb = np.copy(self.M1)
        self.M2_rgb[1, 2] -= 40
        self.d2_rgb = np.copy(self.d1)
        ret, _, _, _, _, self.R_rgb, self.T_rgb, E, F = cv2.stereoCalibrate(
            self.objpoints_rgb_r, self.imgpoints_rgb_right, self.imgpoints_rgb,
            self.M2_rgb, self.d2_rgb, self.M3_scaled, self.d3_scaled, self.img_shape_rgb_scaled,
            criteria=stereocalib_criteria, flags=flags)
        print("~~~~~~ Stereo calibration rgb-left RMS error ~~~~~~~~")
        print(ret)

        # Rectification is only to test the epipolar
        self.R1_rgb, self.R2_rgb, self.P1_rgb, self.P2_rgb, self.Q_rgb, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.M2_rgb,
            self.d2_rgb,
            self.M3_scaled,
            self.d3_scaled,
            self.img_shape_rgb_scaled, self.R_rgb, self.T_rgb)

    def process_images(self, filepath):
        """Read images, detect corners, refine corners, and save data."""
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        # polygon ids of left/right image sets with checkerboard corners.
        self.calib_successes = []

        # images_left = glob.glob(filepath + "/left/*")
        # images_right = glob.glob(filepath + "/right/*")
        images_left = glob.glob(filepath + "/left/*")
        images_right = glob.glob(filepath + "/right/*")
        print("Images left path------------------->")
        print(images_left)
        images_left.sort()
        images_right.sort()

        print("\nAttempting to read images for left camera from dir: " +
              filepath + "/left/")
        print("Attempting to read images for right camera from dir: " +
              filepath + "/right/")

        assert len(
            images_left) != 0, "ERROR: Images not read correctly, check directory"
        assert len(
            images_right) != 0, "ERROR: Images not read correctly, check directory"

        self.temp_img_r_point_list = []
        self.temp_img_l_point_list = []

        for image_left, image_right in zip(images_left, images_right):
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            assert img_l is not None, "ERROR: Images not read correctly"
            assert img_r is not None, "ERROR: Images not read correctly"

            print("Finding chessboard corners for %s and %s..." %
                  (os.path.basename(image_left), os.path.basename(image_right)))
            start_time = time.time()

            # Find the chess board corners
            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
            ret_l, corners_l = cv2.findChessboardCorners(img_l, (9, 6), flags)
            ret_r, corners_r = cv2.findChessboardCorners(img_r, (9, 6), flags)

            # termination criteria
            self.criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                             cv2.TERM_CRITERIA_EPS, 30, 0.001)

            # if corners are found in both images, refine and add data
            if ret_l and ret_r:
                self.objpoints.append(self.objp)
                rt = cv2.cornerSubPix(img_l, corners_l, (5, 5),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)
                rt = cv2.cornerSubPix(img_r, corners_r, (5, 5),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)
                self.temp_img_l_point_list.append([corners_l])
                self.temp_img_r_point_list.append([corners_r])
                # self.calib_successes.append(polygon_from_image_name(image_left))
                print("\t[OK]. Took %i seconds." %
                      (round(time.time() - start_time, 2)))
            else:
                print("\t[ERROR] - Corners not detected. Took %i seconds." %
                      (round(time.time() - start_time, 2)))

            self.img_shape = img_r.shape[::-1]
        print(str(len(self.objpoints)) + " of " + str(len(images_left)) +
              " images being used for calibration")
        # self.ensure_valid_images()

    def ensure_valid_images(self):
        """
        Ensures there is one set of left/right images for each polygon. If not, raises an raises an
        AssertionError with instructions on re-running calibration for the invalid polygons.
        """
        expected_polygons = len(setPolygonCoordinates(
            1000, 600))  # inseted values are placeholders
        unique_calib_successes = set(self.calib_successes)
        if len(unique_calib_successes) != expected_polygons:
            valid = set(np.arange(0, expected_polygons))
            missing = valid - unique_calib_successes
            arg_value = ' '.join(map(str, missing))
            raise AssertionError(
                "Missing valid image sets for %i polygons. Re-run calibration with the\n'-p %s' argument to re-capture images for these polygons." % (len(missing), arg_value))
        else:
            return True

    def calibrate_camera(self):
        """Calibrate camera and construct Homography."""
        # init camera calibrations
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

    def stereo_calib_two_homo(self):
        # config
        flags = 0
        #flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        #flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_RATIONAL_MODEL
        #flags |= cv2.CALIB_FIX_K1
        #flags |= cv2.CALIB_FIX_K2
        #flags |= cv2.CALIB_FIX_K3
        #flags |= cv2.CALIB_FIX_K4
        #flags |= cv2.CALIB_FIX_K5
        #flags |= cv2.CALIB_FIX_K6
        # flags |= cv::CALIB_ZERO_TANGENT_DIST

        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # stereo calibration procedure
        ret, self.M1, self.d1, self.M2, self.d2, self.R, self.T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)
        print("~~~~~~~~~~~~~RMS of stereo ~>")
        print(
            'RMS error of stereo calibration of left-right is {0}'.format(ret))

        self.R1, self.R2, self.P1, self.P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.M1,
            self.d1,
            self.M2,
            self.d2,
            self.img_shape, self.R, self.T)

        self.H1 = np.matmul(np.matmul(self.M2, self.R1),
                            np.linalg.inv(self.M1))
        self.H2 = np.matmul(np.matmul(self.M2, self.R2),
                            np.linalg.inv(self.M2))

    def test_epipolar_checker(self, dataset_dir):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()
        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"

        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)
            # warp right image
            img_l = cv2.warpPerspective(img_l, self.H1, img_l.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            img_r = cv2.warpPerspective(img_r, self.H2, img_r.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            image_data_pairs.append((img_l, img_r))

        # compute metrics
        imgpoints_r = []
        imgpoints_l = []
        for image_data_pair in image_data_pairs:
            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
            flags |= cv2.CALIB_CB_FAST_CHECK
            ret_l, corners_l = cv2.findChessboardCorners(image_data_pair[0],
                                                         (9, 6), flags)
            ret_r, corners_r = cv2.findChessboardCorners(image_data_pair[1],
                                                         (9, 6), flags)

            # termination criteria
            self.criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                             cv2.TERM_CRITERIA_EPS, 10, 0.05)

            # if corners are found in both images, refine and add data
            if ret_l and ret_r:
                rt = cv2.cornerSubPix(image_data_pair[0], corners_l, (5, 5),
                                      (-1, -1), self.criteria)
                rt = cv2.cornerSubPix(image_data_pair[1], corners_r, (5, 5),
                                      (-1, -1), self.criteria)
                imgpoints_l.extend(corners_l)
                imgpoints_r.extend(corners_r)
                epi_error_sum = 0
                for l_pt, r_pt in zip(corners_l, corners_r):
                    epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

                # print("Average Epipolar Error per image on host: " + str(epi_error_sum / len(corners_l)))

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

        avg_epipolar = epi_error_sum / len(imgpoints_r)
        print("Average Epipolar Error: " + str(avg_epipolar))

        return avg_epipolar

        # if avg_epipolar > 0.5:
        #     fail_img = cv2.imread(consts.resource_paths.calib_fail_path, cv2.IMREAD_COLOR)
        #     cv2.imshow('Calibration test Failed', fail_img)
        # else:
        #     self.rundepthai()
        #     if not depthai.init_device(consts.resource_paths.device_cmd_fpath, ''):
        #         print("Error initializing device. Try to reset it.")
        #         exit(1)

        #     if depthai.is_eeprom_loaded():
        #         pass_img = cv2.imread(consts.resource_paths.pass_path, cv2.IMREAD_COLOR)
        #         while (1):
        #             cv2.imshow('Calibration test Passed and wrote to EEPROM', pass_img)
        #             k = cv2.waitKey(33)
        #             if k == 32 or k == 27:  # Esc key to stop
        #                 break
        #             elif k == -1:  # normally -1 returned,so don't print it
        #                 continue
        #     else:
        #         fail_img = cv2.imread(consts.resource_paths.eeprom_fail_path, cv2.IMREAD_COLOR)
        #         while (1):
        #             cv2.imshow('EEPROM write failed', fail_img)
        #             k = cv2.waitKey(33)
        #             if k == 32 or k == 27:  # Esc key to stop
        #                 break
        #             elif k == -1:  # normally -1 returned,so don't print it
        #                 continue
        #     depthai.deinit_device()

    def test_epipolar_charuco(self, dataset_dir):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()
        print("HU IHER")
        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # if not use_homo:
        mapx_l, mapy_l = cv2.initUndistortRectifyMap(
            self.M1, self.d1, self.R1, self.P1, self.img_shape, cv2.CV_32FC1)
        mapx_r, mapy_r = cv2.initUndistortRectifyMap(
            self.M2, self.d2, self.R2, self.P2, self.img_shape, cv2.CV_32FC1)

        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)
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

#             if len(marker_corners_l)>0 and len(marker_corners_r)>0:
#                 for corner in marker_corners_l:
#                     cv2.cornerSubPix(image_data_pair[0], corner,
#                                      winSize = (5,5),
#                                      zeroZone = (-1,-1),
#                                      criteria = criteria)
#                 for corner in marker_corners_r:
#                     cv2.cornerSubPix(image_data_pair[1], corner,
#                                      winSize = (5,5),
#                                      zeroZone = (-1,-1),
#                                      criteria = criteria)
            res2_l = cv2.aruco.interpolateCornersCharuco(
                marker_corners_l, ids_l, image_data_pair[0], self.board)
            res2_r = cv2.aruco.interpolateCornersCharuco(
                marker_corners_r, ids_r, image_data_pair[1], self.board)
            if res2_l[1] is not None and res2_l[2] is not None and len(res2_l[1]) > 3:

                cv2.cornerSubPix(image_data_pair[0], res2_l[1],
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
                cv2.cornerSubPix(image_data_pair[1], res2_r[1],
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)

            # termination criteria
            corners_l = []
            corners_r = []
            for j in range(len(res2_l[2])):
                idx = np.where(res2_r[2] == res2_l[2][j])
                if idx[0].size == 0:
                    continue
                corners_l.append(res2_l[1][j])
                corners_r.append(res2_r[1][idx])
#                 obj_pts_sub.append(one_pts[allIds_l[i][j]])

#             obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
#             left_sub_corners_sampled.append(np.array(left_sub_corners, dtype=np.float32))
#             right_sub_corners_sampled.append(np.array(right_sub_corners, dtype=np.float32))

            imgpoints_l.extend(corners_l)
            imgpoints_r.extend(corners_r)
            epi_error_sum = 0
            for l_pt, r_pt in zip(corners_l, corners_r):
                epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])
            img_pth = Path(images_right[i])
            name = img_pth.name
            print("Average Epipolar Error per image on host in " + img_pth.name + " : " +
                  str(epi_error_sum / len(corners_l)))

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

        avg_epipolar = epi_error_sum / len(imgpoints_r)
        print("Average Epipolar Error: " + str(avg_epipolar))

        return avg_epipolar

    def test_epipolar_charuco_rgb(self, dataset_dir):
        images_rgb = glob.glob(dataset_dir + '/rgb/*.png')
        images_right = glob.glob(dataset_dir + '/left/*.png')
        images_rgb.sort()
        images_right.sort()
        print("HU IHER")
        assert len(images_rgb) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"
        # criteria for marker detection/corner detections
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        scale_width = 1280/self.img_shape_rgb[0]
        print('scaled using {0}'.format(self.img_shape_rgb[0]))

        # if not use_homo:
        mapx_rgb, mapy_rgb = cv2.initUndistortRectifyMap(
            self.M3_scaled, self.d3_scaled, self.R2_rgb, self.P1_rgb, self.img_shape_rgb_scaled, cv2.CV_32FC1)
        mapx_r, mapy_r = cv2.initUndistortRectifyMap(
            self.M2_rgb, self.d2_rgb, self.R1_rgb, self.P2_rgb, self.img_shape_rgb_scaled, cv2.CV_32FC1)

        # self.H1_rgb = np.matmul(np.matmul(self.M2, self.R1_rgb),
        #                     np.linalg.inv(M_rgb))
        # self.H2_r = np.matmul(np.matmul(self.M2, self.R2_rgb),
        #                     np.linalg.inv(self.M2))

        image_data_pairs = []
        count = 0
        for image_rgb, image_right in zip(images_rgb, images_right):
            # read images
            img_rgb = cv2.imread(image_rgb, 0)
            img_r = cv2.imread(image_right, 0)
            img_r = img_r[40: 760, :]
            # TODO(sachin): replace Hard coded scaleratio for rgb resize
            dest_res = (int(img_rgb.shape[1] * scale_width),
                        int(img_rgb.shape[0] * scale_width))
            img_rgb = cv2.resize(
                img_rgb, dest_res, interpolation=cv2.INTER_CUBIC)
            if img_rgb.shape[0] < 720:
                raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
                    img_rgb.shape[0], req_resolution[0]))
            del_height = (img_rgb.shape[0] - 720) // 2
            print("del height ??")
            print(del_height)
            img_rgb = img_rgb[del_height: del_height + 720, :]
            print("resized_shape")
            print(img_rgb.shape)
            # self.parse_frame(img_rgb, "rectified_rgb_before",
            #                  "rectified_"+str(count))

            # warp right image

            # img_rgb = cv2.warpPerspective(img_rgb, self.H1_rgb, img_rgb.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            # img_r = cv2.warpPerspective(img_r, self.H2_r, img_r.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            img_rgb = cv2.remap(img_rgb, mapx_rgb, mapy_rgb, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)
            # self.parse_frame(img_rgb, "rectified_rgb", "rectified_"+str(count))
            image_data_pairs.append((img_rgb, img_r))
            count += 1

        # here anything with _l represents rgb camera
        # TODO(sachin): change it to _rgb once everything is stable
        # compute metrics
        imgpoints_r = []
        imgpoints_l = []
        for image_data_pair in image_data_pairs:
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

#             if len(marker_corners_l)>0 and len(marker_corners_r)>0:
#                 for corner in marker_corners_l:
#                     cv2.cornerSubPix(image_data_pair[0], corner,
#                                      winSize = (5,5),
#                                      zeroZone = (-1,-1),
#                                      criteria = criteria)
#                 for corner in marker_corners_r:
#                     cv2.cornerSubPix(image_data_pair[1], corner,
#                                      winSize = (5,5),
#                                      zeroZone = (-1,-1),
#                                      criteria = criteria)
            res2_l = cv2.aruco.interpolateCornersCharuco(
                marker_corners_l, ids_l, image_data_pair[0], self.board)
            res2_r = cv2.aruco.interpolateCornersCharuco(
                marker_corners_r, ids_r, image_data_pair[1], self.board)
            if res2_l[1] is not None and res2_l[2] is not None and len(res2_l[1]) > 3:

                cv2.cornerSubPix(image_data_pair[0], res2_l[1],
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
                cv2.cornerSubPix(image_data_pair[1], res2_r[1],
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)

            # termination criteria
            corners_l = []
            corners_r = []
            for j in range(len(res2_l[2])):
                idx = np.where(res2_r[2] == res2_l[2][j])
                if idx[0].size == 0:
                    continue
                corners_l.append(res2_l[1][j])
                corners_r.append(res2_r[1][idx])
#                 obj_pts_sub.append(one_pts[allIds_l[i][j]])

#             obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
#             left_sub_corners_sampled.append(np.array(left_sub_corners, dtype=np.float32))
#             right_sub_corners_sampled.append(np.array(right_sub_corners, dtype=np.float32))

            imgpoints_l.extend(corners_l)
            imgpoints_r.extend(corners_r)
            epi_error_sum = 0
            for l_pt, r_pt in zip(corners_l, corners_r):
                epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

            print("Average Epipolar Error per image on host on rgb_right: " +
                  str(epi_error_sum / len(corners_l)))

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

        avg_epipolar = epi_error_sum / len(imgpoints_r)
        print("Average Epipolar Error of rgb_right: " + str(avg_epipolar))

        return avg_epipolar

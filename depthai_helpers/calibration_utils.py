import cv2
import glob
import os
import shutil
import numpy as np
import re
import time
import consts.resource_paths
import json

# Creates a set of 13 polygon coordinates
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

# Filters polygons to just those at the given indexes.
def select_polygon_coords(p_coordinates,indexes):
    if indexes == None:
        # The default
        return p_coordinates
    else:
        print("Filtering polygons to those at indexes=",indexes)
        return [p_coordinates[i] for i in indexes]

def image_filename(stream_name,polygon_index,total_num_of_captured_images):
    return "{stream_name}_p{polygon_index}_{total_num_of_captured_images}.png".format(stream_name=stream_name,polygon_index=polygon_index,total_num_of_captured_images=total_num_of_captured_images)

def polygon_from_image_name(image_name):
    """Returns the polygon index from an image name (ex: "left_p10_0.png" => 10)"""
    return int(re.findall("p(\d+)",image_name)[0])

class StereoCalibration(object):
    """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def __init__(self):
        """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def calibrate(self, filepath, square_size, out_filepath, flags):
        """Function to calculate calibration for stereo camera."""
        start_time = time.time()
        # init object data
        self.objp = np.zeros((9 * 6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        for pt in self.objp:
            pt *= square_size

        # process images, detect corners, refine and save data
        self.process_images(filepath)

        # run calibration procedure and construct Homography and mesh
        self.stereo_calibrate_two_homography_calib()
        # save data to binary file

        R1_fp32 = self.R1.astype(np.float32)
        R2_fp32 = self.R2.astype(np.float32)
        M1_fp32 = self.M1.astype(np.float32)
        M2_fp32 = self.M2.astype(np.float32)
        R_fp32  = self.R.astype(np.float32)
        T_fp32  = self.T.astype(np.float32)
        M3_fp32 = np.zeros((3, 3), dtype = np.float32)
        R_rgb_fp32 = np.zeros((3, 3), dtype = np.float32) 
        T_rgb_fp32 = np.zeros(3, dtype = np.float32)  
        d1_coeff_fp32 = self.d1.astype(np.float32)
        d2_coeff_fp32 = self.d2.astype(np.float32)
        d3_coeff_fp32 = np.zeros(14, dtype = np.float32)

        with open(out_filepath, "wb") as fp:
            fp.write(R1_fp32.tobytes()) # goes to left camera
            fp.write(R2_fp32.tobytes()) # goes to right camera
            fp.write(M1_fp32.tobytes()) # left camera intrinsics
            fp.write(M2_fp32.tobytes()) # right camera intrinsics
            fp.write(R_fp32.tobytes()) # Rotation matrix left -> right
            fp.write(T_fp32.tobytes()) # Translation vector left -> right
            fp.write(M3_fp32.tobytes()) # rgb camera intrinsics ## Currently a zero matrix
            fp.write(R_rgb_fp32.tobytes()) # Rotation matrix left -> rgb ## Currently Identity matrix
            fp.write(T_rgb_fp32.tobytes()) # Translation vector left -> rgb ## Currently vector of zeros
            fp.write(d1_coeff_fp32.tobytes()) # distortion coeff of left camera
            fp.write(d2_coeff_fp32.tobytes()) # distortion coeff of right camera
            fp.write(d3_coeff_fp32.tobytes()) # distortion coeff of rgb camera - currently zeros

        if 0: # Print matrices, to compare with device data
            np.set_printoptions(suppress=True, precision=6)
            print("\nR1 (left)");  print(R1_fp32)
            print("\nR2 (right)"); print(R2_fp32)
            print("\nM1 (left)");  print(M1_fp32)
            print("\nM2 (right)"); print(M2_fp32)
            print("\nR");          print(R_fp32)
            print("\nT");          print(T_fp32)
            print("\nM3 (rgb)");   print(M3_fp32)

        if 0: # Print computed homography, to compare with device data
            np.set_printoptions(suppress=True, precision=6)
            for res_height in [800, 720, 400]:
                m1 = np.copy(M1_fp32)
                m2 = np.copy(M2_fp32)
                if res_height == 720:
                    m1[1,2] -= 40
                    m2[1,2] -= 40
                if res_height == 400:
                    m_scale = [[0.5,   0, 0],
                               [  0, 0.5, 0],
                               [  0,   0, 1]]
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
        with open(out_filepath, "ab") as fp:
            fp.write(bytearray(flags))

        print("Calibration file written to %s." % (out_filepath))
        print("\tTook %i seconds to run image processing." % (round(time.time() - start_time, 2)))
        # show debug output for visual inspection
        print("\nRectifying dataset for visual inspection using Mesh")
        self.show_rectified_images_two_calib(filepath, False)
        print("\nRectifying dataset for visual inspection using Two Homography")
        self.show_rectified_images_two_calib(filepath, True)

    def process_images(self, filepath):
        """Read images, detect corners, refine corners, and save data."""
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        self.calib_successes = [] # polygon ids of left/right image sets with checkerboard corners.

        images_left = glob.glob(filepath + "/left/*")
        images_right = glob.glob(filepath + "/right/*")
        images_left.sort()
        images_right.sort()

        print("\nAttempting to read images for left camera from dir: " +
              filepath + "/left/")
        print("Attempting to read images for right camera from dir: " +
              filepath + "/right/")

        assert len(images_left) != 0, "ERROR: Images not read correctly, check directory"
        assert len(images_right) != 0, "ERROR: Images not read correctly, check directory"

        self.temp_img_r_point_list = []
        self.temp_img_l_point_list = []

        for image_left, image_right in zip(images_left, images_right):
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)


            assert img_l is not None, "ERROR: Images not read correctly"
            assert img_r is not None, "ERROR: Images not read correctly"

            print("Finding chessboard corners for %s and %s..." % (os.path.basename(image_left), os.path.basename(image_right)))
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
                self.calib_successes.append(polygon_from_image_name(image_left))
                print("\t[OK]. Took %i seconds." % (round(time.time() - start_time, 2)))
            else:
                print("\t[ERROR] - Corners not detected. Took %i seconds." % (round(time.time() - start_time, 2)))

            self.img_shape = img_r.shape[::-1]
        print(str(len(self.objpoints)) + " of " + str(len(images_left)) +
              " images being used for calibration")
        self.ensure_valid_images()

    def ensure_valid_images(self):
        """
        Ensures there is one set of left/right images for each polygon. If not, raises an raises an
        AssertionError with instructions on re-running calibration for the invalid polygons.
        """
        expected_polygons = len(setPolygonCoordinates(1000,600)) # inseted values are placeholders
        unique_calib_successes = set(self.calib_successes)
        if len(unique_calib_successes) != expected_polygons:
            valid = set(np.arange(0,expected_polygons))
            missing = valid - unique_calib_successes
            arg_value = ' '.join(map(str, missing))
            raise AssertionError("Missing valid image sets for %i polygons. Re-run calibration with the\n'-p %s' argument to re-capture images for these polygons." % (len(missing), arg_value))
        else:
            return True
    
    def stereo_calibrate(self):
        """Calibrate camera and construct Homography."""
        # init camera calibrations
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

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
        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # stereo calibration procedure
        ret, self.M1, self.d1, self.M2, self.d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)

        assert ret < 1.0, "[ERROR] Calibration RMS error < 1.0 (%i). Re-try image capture." % (ret)
        print("[OK] Calibration successful w/ RMS error=" + str(ret))

        # construct Homography
        plane_depth = 40000000.0  # arbitrary plane depth 
        #TODO: Need to understand effect of plane_depth. Why does this improve some boards' cals?
        self.H = np.dot(self.M2, np.dot(R, np.linalg.inv(self.M1)))
        self.H /= self.H[2, 2]
        # rectify Homography for right camera
        disparity = (self.M1[0, 0] * T[0] / plane_depth)
        self.H[0, 2] -= disparity
        self.H = self.H.astype(np.float32)
        print("Rectifying Homography...")
        print(self.H)

    def stereo_calibrate_two_homography_calib(self):
        """Calibrate camera and construct Homography."""
        # init camera calibrations
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

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
        #flags |= cv::CALIB_ZERO_TANGENT_DIST

        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # stereo calibration procedure
        ret, self.M1, self.d1, self.M2, self.d2, self.R, self.T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)

        self.R1, self.R2, self.P1, self.P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                                                                                                self.M1,
                                                                                                self.d1,
                                                                                                self.M2,
                                                                                                self.d2,
                                                                                                self.img_shape, self.R, self.T)

        self.H1 = np.matmul(np.matmul(self.M2, self.R1), np.linalg.inv(self.M1))
        self.H2 = np.matmul(np.matmul(self.M2, self.R2), np.linalg.inv(self.M2))                                                                                        


    def create_save_mesh(self): #, output_path):

        map_x_l, map_y_l = cv2.initUndistortRectifyMap(self.M1, self.d1, self.R1, self.M2, self.img_shape, cv2.CV_32FC1)
        map_x_r, map_y_r = cv2.initUndistortRectifyMap(self.M2, self.d2, self.R2, self.M2, self.img_shape, cv2.CV_32FC1)
        print("Distortion coeff left cam")
        print(self.d1)
        print("Distortion coeff right cam ")
        print(self.d2)

        # print(str(type(map_x_l)))
        # map_x_l.tofile(consts.resource_paths.left_mesh_fpath)
        # map_y_l.tofile(out_filepath)
        # map_x_r.tofile(out_filepath)
        # map_y_r.tofile(out_filepath)

        map_x_l_fp32 = map_x_l.astype(np.float32)
        map_y_l_fp32 = map_y_l.astype(np.float32)
        map_x_r_fp32 = map_x_r.astype(np.float32)
        map_y_r_fp32 = map_y_r.astype(np.float32)

        with open(consts.resource_paths.left_mesh_fpath, "ab") as fp:
            fp.write(map_x_l_fp32.tobytes())
            fp.write(map_y_l_fp32.tobytes())
        
        with open(consts.resource_paths.right_mesh_fpath, "ab") as fp:    
            fp.write(map_x_r_fp32.tobytes())
            fp.write(map_y_r_fp32.tobytes())
        
        print("shape of maps")
        print(map_x_l.shape)
        print(map_y_l.shape)
        print(map_x_r.shape)
        print(map_y_r.shape)
        


        # meshCellSize = 16
        # mesh_left = []
        # mesh_right = []

        # for y in range(map_x_l.shape[0] + 1):
        #     if y % meshCellSize == 0:
        #         row_left = []
        #         row_right = []
        #         for x in range(map_x_l.shape[1] + 1):
        #             if x % meshCellSize == 0:
        #                 if y == map_x_l.shape[0] and x == map_x_l.shape[1]:
        #                     row_left.append(map_y_l[y - 1, x - 1])
        #                     row_left.append(map_x_l[y - 1, x - 1])
        #                     row_right.append(map_y_r[y - 1, x - 1])
        #                     row_right.append(map_x_r[y - 1, x - 1])
        #                 elif y == map_x_l.shape[0]:
        #                     row_left.append(map_y_l[y - 1, x])
        #                     row_left.append(map_x_l[y - 1, x])
        #                     row_right.append(map_y_r[y - 1, x])
        #                     row_right.append(map_x_r[y - 1, x])
        #                 elif x == map_x_l.shape[1]:
        #                     row_left.append(map_y_l[y, x - 1])
        #                     row_left.append(map_x_l[y, x - 1])
        #                     row_right.append(map_y_r[y, x - 1])
        #                     row_right.append(map_x_r[y, x - 1])
        #                 else:
        #                     row_left.append(map_y_l[y, x])
        #                     row_left.append(map_x_l[y, x])
        #                     row_right.append(map_y_r[y, x])
        #                     row_right.append(map_x_r[y, x])
        #         if (map_x_l.shape[1] % meshCellSize) % 2 != 0:
        #                     row_left.append(0)
        #                     row_left.append(0)
        #                     row_right.append(0)
        #                     row_right.append(0)

        #         mesh_left.append(row_left)
        #         mesh_right.append(row_right)    
        
        # mesh_left = np.array(mesh_left)
        # mesh_right = np.array(mesh_right)
        # mesh_left.tofile(consts.resource_paths.left_mesh_fpath)
        # mesh_right.tofile(consts.resource_paths.right_mesh_fpath)

    def show_rectified_images_two_calib(self, dataset_dir, use_homo):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()
        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"

        if not use_homo:
            mapx_l, mapy_l = cv2.initUndistortRectifyMap(self.M1, self.d1, self.R1, self.P1, self.img_shape, cv2.CV_32FC1)
            mapx_r, mapy_r = cv2.initUndistortRectifyMap(self.M2, self.d2, self.R2, self.P2, self.img_shape, cv2.CV_32FC1)
            # mapx_l, mapy_l = self.rectify_map(self.M1, self.d1[0], self.R1)
            # mapx_r, mapy_r = self.rectify_map(self.M2, self.d2[0], self.R2)

        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)
            if not use_homo:
                img_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
                img_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)
            else:
                # img_l = cv2.undistort(img_l, self.M1, self.d1, None, self.M1)
                # img_r = cv2.undistort(img_r, self.M2, self.d2, None, self.M2)

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
                # epi_error_sum = 0
                # for l_pt, r_pt in zip(corners_l, corners_r):
                #     epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])
                
                # print("Average Epipolar Error per image on host: " + str(epi_error_sum / len(corners_l)))

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

        print("Average Epipolar Error: " + str(epi_error_sum / len(imgpoints_r)))

        for image_data_pair in image_data_pairs:
            img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)

            # draw epipolar lines for debug purposes
            line_row = 0
            while line_row < img_concat.shape[0]:
                cv2.line(img_concat,
                         (0, line_row), (img_concat.shape[1], line_row),
                         (0, 255, 0), 1)
                line_row += 30

            # show image
            print("Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
            while (1):
                cv2.imshow('Stereo Pair', img_concat)
                k = cv2.waitKey(33)
                if k == 32:  # Esc key to stop
                    break
                elif k == 27:
                    os._exit(0)
                    # raise SystemExit()
                elif k == -1:  # normally -1 returned,so don't print it
                    continue


    def rectify_map(self, M, d, R):
        fx = M[0,0]
        fy = M[1,1]
        u0 = M[0,2]
        v0 = M[1,2]
        # distortion coefficients
        print("distortion coeff")
        print(d)
        print(d.shape)
        k1 = d[0]
        k2 = d[1] 
        p1 = d[2] 
        p2 = d[3] 
        k3 = d[4] 
        s4 = d[11]
        k4 = d[5]
        k5 = d[6]
        k6 = d[7]
        s1 = d[8]
        s2 = d[9] 
        s3 = d[10]
        tauX = d[12] 
        tauY = d[13]


        
        matTilt = np.identity(3, dtype=np.float32)
        ir = np.linalg.inv(np.matmul(self.M2, R)) ## Change it to using LU
        # s_x = [] ## NOT USED
        # s_y = []
        # s_w = []
        # for i in range(8):
        #     s_x.append(ir[0,0] * i)
        #     s_y.append(ir[1,0] * i)
        #     s_w.append(ir[2,0] * i)
        map_x = np.zeros((800,1280),dtype=np.float32)
        map_y = np.zeros((800,1280),dtype=np.float32)

        for i in range(800):
            _x = i * ir[0,1] + ir[0,2]
            _y = i * ir[1,1] + ir[1,2]
            _w = i * ir[2,1] + ir[2,2]

            for j in range(1280):
                _x += ir[0,0]
                _y += ir[1,0]
                _w += ir[2,0]
                w = 1/_w
                x = _x * w
                y = _y * w

                x2 = x*x
                y2 = y*y
                r2 = x2  + y2
                _2xy = 2*x*y
                kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2)/(1 + ((k6*r2 + k5)*r2 + k4)*r2)
                xd = (x*kr + p1*_2xy + p2*(r2 + 2*x2) + s1*r2+s2*r2*r2)
                yd = (y*kr + p1*(r2 + 2*y2) + p2*_2xy + s3*r2+s4*r2*r2)
                vec_3d = np.array([xd,yd,1]).reshape(3,1)
                vecTilt = np.matmul(matTilt, vec_3d);
                # Vec3d vecTilt = matTilt*cv::Vec3d(xd, yd, 1);
                invProj = 1./vecTilt[2] if vecTilt[2] else 1
                # double invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
                # double u = fx*invProj*vecTilt(0) + u0;
                # double v = fy*invProj*vecTilt(1) + v0;    
                u = fx*invProj*vecTilt[0] + u0; # u0 and v0 is from the M1
                v = fy*invProj*vecTilt[1] + v0;                
                map_x[i,j] = u
                map_y[i,j] = v
        print("map built")
        return map_x, map_y


    def stereo_calibrate_two_homography_uncalib(self):
        """Calibrate camera and construct Homography."""
        # init camera calibrations
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

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
        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # stereo calibration procedure
        ret, self.M1, self.d1, self.M2, self.d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)

        assert ret < 1.0, "[ERROR] Calibration RMS error < 1.0 (%i). Re-try image capture." % (ret)
        print("[OK] Calibration successful w/ RMS error=" + str(ret))

        F_test, mask = cv2.findFundamentalMat(np.array(self.imgpoints_l).reshape(-1, 2),
                                   np.array(self.imgpoints_r).reshape(-1, 2),
                                   cv2.FM_RANSAC, 2) # try ransac and other methods too.

        res, self.H1, self.H2 = cv2.stereoRectifyUncalibrated(np.array(self.imgpoints_l).reshape(-1, 2),
                                                    np.array(self.imgpoints_r).reshape(-1, 2),
                                                    F_test,
                                                    self.img_shape, 2)

    def test_img_vis(self, dataset_dir):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()

        map1_l, map2_l = cv2.initUndistortRectifyMap(self.M1, self.d1, self.R1, self.M1, self.img_shape, cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(self.M2, self.d2, self.R2, self.M2, self.img_shape, cv2.CV_32FC1)

        map1_l_review, map2_l_review = self.rectify_map(self.M1, self.d1[0], self.R1)
        map1_r_review, map2_r_review = self.rectify_map(self.M2, self.d2[0], self.R2)

        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):

            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            img_l_cv2 = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_CUBIC)
            img_r_cv2 = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_CUBIC)

            img_l_review = cv2.remap(img_l, map1_l_review, map2_l_review, cv2.INTER_CUBIC)
            img_r_review = cv2.remap(img_r, map1_r_review, map2_r_review, cv2.INTER_CUBIC)
            img_concat = cv2.hconcat([img_l_review, img_l_cv2])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)
            while(1):
                cv2.imshow('Stereo Pair', img_concat)
                k = cv2.waitKey(33)
                if k == 32:    
                    break
                elif k == 27: # Esc key to stop
                    break
                    # raise SystemExit()
                elif k == -1:  # normally -1 returned,so don't print it
                    continue
            img_concat = cv2.hconcat([img_r_review, img_r_cv2])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)
            while(1):
                cv2.imshow('Stereo Pair', img_concat)
                k = cv2.waitKey(33)
                if k == 32:    
                    break
                elif k == 27: # Esc key to stop
                    break
                    # raise SystemExit()
                elif k == -1:  # normally -1 returned,so don't print it
                    continue


    def show_rectified_images_two_uncalib(self, dataset_dir):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()

        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"

        # H = np.fromfile(calibration_file, dtype=np.float32).reshape((3, 3))

        # print("Using Homography from file, with values: ")
        # print(H)

        # H1 = np.linalg.inv(H1)
        # H2 = np.linalg.inv(H2)
        R1 = np.matmul(np.matmul(np.linalg.inv(self.M1), self.H1), self.M1)
        R2 = np.matmul(np.matmul(np.linalg.inv(self.M2), self.H2), self.M2)
        img_l = cv2.imread(images_left[0], 0)
        map1_l, map2_l = cv2.initUndistortRectifyMap(self.M1, self.d1, R1, self.M1, self.img_shape, cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(self.M2, self.d2, R2, self.M2, self.img_shape, cv2.CV_32FC1)



        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)
            # img_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_CUBIC)
            # img_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_CUBIC)

            # cv2.initUndistortRectifyMap(self.M1)


            # # warp right image
            img_l = cv2.warpPerspective(img_l, H1, img_l.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            img_r = cv2.warpPerspective(img_r, H2, img_r.shape[::-1],
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
            ret_l, corners_l = cv2.findChessboardCorners(image_data_pair[0],
                                                         (9, 6), flags)
            ret_r, corners_r = cv2.findChessboardCorners(image_data_pair[1],
                                                         (9, 6), flags)

            # termination criteria
            self.criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                             cv2.TERM_CRITERIA_EPS, 30, 0.001)

            # if corners are found in both images, refine and add data
            if ret_l and ret_r:
                rt = cv2.cornerSubPix(image_data_pair[0], corners_l, (5, 5),
                                      (-1, -1), self.criteria)
                rt = cv2.cornerSubPix(image_data_pair[1], corners_r, (5, 5),
                                      (-1, -1), self.criteria)
                imgpoints_l.extend(corners_l)
                imgpoints_r.extend(corners_r)

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

        print("Average Epipolar Error: " + str(epi_error_sum / len(imgpoints_r)))

        for image_data_pair in image_data_pairs:
            img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)

            # draw epipolar lines for debug purposes
            line_row = 0
            while line_row < img_concat.shape[0]:
                cv2.line(img_concat,
                         (0, line_row), (img_concat.shape[1], line_row),
                         (0, 255, 0), 1)
                line_row += 30

            # show image
            print("Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
            while (1):
                cv2.imshow('Stereo Pair', img_concat)
                k = cv2.waitKey(33)
                if k == 32:  # Esc key to stop
                    break
                elif k == 27:
                    os._exit(0)
                    # raise SystemExit()
                elif k == -1:  # normally -1 returned,so don't print it
                    continue






    def show_rectified_images(self, dataset_dir, calibration_file):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()

        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"

        H = np.fromfile(calibration_file, dtype=np.float32).reshape((3, 3))

        print("Using Homography from file, with values: ")
        print(H)

        H = np.linalg.inv(H)
        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            # warp right image
            img_r = cv2.warpPerspective(img_r, H, img_r.shape[::-1],
                                        cv2.INTER_LINEAR +
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
    

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

        print("Average Epipolar Error: " + str(epi_error_sum / len(imgpoints_r)))

        for image_data_pair in image_data_pairs:
            img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)

            # draw epipolar lines for debug purposes
            line_row = 0
            while line_row < img_concat.shape[0]:
                cv2.line(img_concat,
                         (0, line_row), (img_concat.shape[1], line_row),
                         (0, 255, 0), 1)
                line_row += 30

            # show image
            print("Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
            while(1):
                cv2.imshow('Stereo Pair', img_concat)
                k = cv2.waitKey(33)
                if k == 32:    # Esc key to stop
                    break
                elif k == 27:
                    os._exit(0)
                    # raise SystemExit()
                elif k == -1:  # normally -1 returned,so don't print it
                    continue

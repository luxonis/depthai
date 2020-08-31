import cv2
import glob
import os
import shutil
import numpy as np
import re
import time

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

        # run calibration procedure and construct Homography
        # self.stereo_calibrate()
        # H1, H2 = self.stereo_calibrate_two_homography_uncalib()
        self.stereo_calibrate_two_homography_calib()
        # save data to binary file
        # self.H.tofile(out_filepath)

        with open("out_filepath", "wb") as fp:
            fp.write(self.H1.tobytes())
            fp.write(self.H2.tobytes())
            fp.write(self.M1.tobytes())
            fp.write(self.M2.tobytes())
            fp.write(self.R.tobytes())
            fp.write(self.T.tobytes())

        self.create_save_mesh()

        # append specific flags to file
        with open(out_filepath, "ab") as fp:
            fp.write(bytearray(flags))

        print("Calibration file written to %s." % (out_filepath))
        print("\tTook %i seconds to run image processing." % (round(time.time() - start_time, 2)))
        # show debug output for visual inspection
        print("\nRectifying dataset for visual inspection")
        self.show_rectified_images_two_calib(filepath)
        # self.show_rectified_images_two(filepath, H1, H2)
        self.show_rectified_images(filepath, out_filepath)

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
        # n = np.array([[0.0], [0.0], [-1.0]])
        # d_inv = 1.0 / plane_depth
        # z = np.dot(T, n.transpose())
        # H = (R - d_inv * z)
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
        # TODO: Try Building mesh from homography later.
        self.H1 = np.matmul(np.matmul(self.M1, self.R1), np.linalg.inv(self.M1))
        self.H2 = np.matmul(np.matmul(self.M1, self.R2), np.linalg.inv(self.M2))                                                                                        


    def create_save_mesh(self): #, output_path):
        
        # TODO: Try replacing P1 and P2 with M2 later and test it.

        map_x_l, map_y_l = cv2.initUndistortRectifyMap(self.M1, self.d1, self.R1, self.P1, self.img_shape, cv2.CV_32FC1)
        map_x_r, map_y_r = cv2.initUndistortRectifyMap(self.M2, self.d2, self.R2, self.P2, self.img_shape, cv2.CV_32FC1)
        print("type of maps built -------->")
        print(str(type(map_x_l)))
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
        
        mesh_left.tofile("mesh_left.calib")
        mesh_right.tofile("mesh_right.calib")

    def show_rectified_images_two_calib(self, dataset_dir):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()

        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"

        map1_l, map2_l = cv2.initUndistortRectifyMap(self.M1, self.d1, self.R1, self.P1, self.img_shape, cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(self.M2, self.d2, self.R2, self.P2, self.img_shape, cv2.CV_32FC1)

        H1 = np.matmul(np.matmul(self.M1, self.R1), np.linalg.inv(self.M1))
        H2 = np.matmul(np.matmul(self.M1, self.R2), np.linalg.inv(self.M2))

        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)
            img_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_CUBIC)
            img_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_CUBIC)

            # img_l = cv2.undistort(img_l, self.M1, self.d1, None, self.M1)
            # img_r = cv2.undistort(img_r, self.M2, self.d2, None, self.M2)

            # # warp right image
            # img_l = cv2.warpPerspective(img_l, H1, img_l.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)
            #
            # img_r = cv2.warpPerspective(img_r, H2, img_r.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)


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
        # s_b = [[0,    -T[2],  T[1]],
        #        [T[2],   0,   -T[0]],
        #        [-T[1], T[0],    0]
        #       ]
        # f_mat = np.matmul(np.matmul(np.matmul(np.transpose(np.linalg.inv(self.M1)), s_b), np.transpose(R)), np.linalg.inv(self.M2))
        # f_mat /= f_mat[2, 2]
        # i = 0
        # for pts_l, pts_r in zip(self.imgpoints_l, self.imgpoints_r):
        #     print("Image number--->" + str(i))
        #     m = pts_l[i]
        #     pt_r = pts_r[i,0]
        #     # pt_r.insert(1)
        #     pt_r = np.append(pt_r, 1).reshape(3, 1)
        #     pt_l = pts_l[i, 0]
        #     # pt_l.append(1)
        #     pt_l = np.append(pt_l, 1).reshape(1, 3)
        #     # print(pt_r)
        #     # print(np.transpose(pt_r) * f_mat * pt_l)
        #     # x = pt_r * f_mat * pt_l
        #     # z = pt_l * f_mat * pt_r
        #
        #     # x = np.dot(np.dot(pt_r, f_mat),pt_l)
        #     # print("res->" + str(x))
        #     y = np.dot(np.dot(pt_l, f_mat),pt_r)
        #     z = np.dot(np.dot(pt_l, F), pt_r)
        #
        #     print("res->y" + str(y) + " --- " + str(z))
        #
        #     i += 1
        # f = f_mat.reshape(3, 3)

        # f = []
        # for row in f_mat:
        #     c = []
        #     for col in row:
        #         c.append(col[0])
        #     f.append(c)
        # f = np.array(f)
        # f /= f[2,2]
        # val_list = np.array(self.imgpoints_l).reshape(-1,2)  #.reshape(54,2)
        F_test, mask = cv2.findFundamentalMat(np.array(self.imgpoints_l).reshape(-1, 2),
                                   np.array(self.imgpoints_r).reshape(-1, 2),
                                   cv2.FM_RANSAC, 2) # try ransac and other methods too.

        res, self.H1, self.H2 = cv2.stereoRectifyUncalibrated(np.array(self.imgpoints_l).reshape(-1, 2),
                                                    np.array(self.imgpoints_r).reshape(-1, 2),
                                                    F_test,
                                                    self.img_shape, 2)


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

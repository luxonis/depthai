import cv2
import glob
import os
import shutil
import numpy as np
from argparse import ArgumentParser


def mkdir_overwrite(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", action="store",
                        type=str, dest="dataset_dir", required=True,
                        help="Path to calibration dataset, NB must conatin /left and /right dirs")
    parser.add_argument("-s", "--square_size_cm", action="store",
                        type=float, dest="square_size_cm", required=False,
                        help="Sqaure size of calibration pattern used, NB Please use [cm]")
    parser.add_argument("-c", "--calib_file", action="store",
                        type=str, dest="calib_filepath", default="./calibration.bin",
                        help="output filepath for calibration")
    parser.add_argument("-a", "--apply_calibration", action="store_true",
                        dest="apply_calibration",
                        help="Instead of calibrating, aplly calibration file to dataset")

    options = parser.parse_args()

    if options.apply_calibration is False and options.square_size_cm is None:
        print("\nError: Attempting to calibrate but no square size provided.\n")
        parser.print_help()
        raise SystemExit()
    elif options.apply_calibration is True and options.square_size_cm is not None:
        print("\nWarning: Sqaure size arg not used.\n")

    return options


class StereoCalibration(object):
    """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def __init__(self):
        """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def calibrate(self, filepath, square_size, out_filepath):
        """Function to calculate calibration for stereo camera."""
        # init object data
        self.objp = np.zeros((9 * 6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        for pt in self.objp:
            pt *= square_size

        # process images, detect corners, refine and save data
        self.process_images(filepath)

        # run calibration procedure and construct Homography
        self.stereo_calibrate()

        # save data to binary file
        self.H.tofile(out_filepath)
        print("Result written to: " + out_filepath)

        # show debug output for visual inspection
        print("\nRectifying dataset for visual inspection")
        self.show_rectified_images(filepath, out_filepath)

    def process_images(self, filepath):
        """Read images, detect corners, refine corners, and save data."""
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

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

        for image_left, image_right in zip(images_left, images_right):
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            assert img_l is not None, "ERROR: Images not read correctly"
            assert img_r is not None, "ERROR: Images not read correctly"

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
            else:
                print("Corners not detected for",
                      str(os.path.basename(image_left)),
                      str(os.path.basename(image_right)))

            self.img_shape = img_r.shape[::-1]
        print(str(len(self.objpoints)) + " of " + str(len(images_left)) +
              " images being used for calibration")
        assert len(self.objpoints) > 4, "ERROR: Not enough valid image sets, please re-capture"

    def stereo_calibrate(self):
        """Calibrate camera and construct Homography."""
        # init camera calibrations
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

        # config
        flags = 0
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_RATIONAL_MODEL
        flags |= cv2.CALIB_FIX_K1
        flags |= cv2.CALIB_FIX_K2
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_FIX_K6
        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # stereo calibration procedure
        ret, self.M1, self.d1, self.M2, self.d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)
	
        assert ret < 1.0, "ERROR: Calibration no succesfull, please re-capture"
        print("Calibration successful, RMS error: " + str(ret))

        # construct Homography
        plane_depth = 40.0  # arbitrary plane depth
        n = np.array([[0.0], [0.0], [-1.0]])
        d_inv = 1.0 / plane_depth
        H = (R - d_inv * np.dot(T, n.transpose()))
        self.H = np.dot(self.M2, np.dot(H, np.linalg.inv(self.M1)))
        self.H /= self.H[2, 2]
        # rectify Homography for right camera
        disparity = (self.M1[0, 0] * T[0] / plane_depth)
        self.H[0, 2] -= disparity
        self.H = self.H.astype(np.float32)
        print("Rectifying Homography:")
        print(self.H)

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
            while(1):
                cv2.imshow('Stereo Pair', img_concat)
                k = cv2.waitKey(33)
                if k == 32:    # Esc key to stop
                    break
                elif k == 27:
                    raise SystemExit()
                elif k == -1:  # normally -1 returned,so don't print it
                    continue

    def rectify_dataset(self, dataset_dir, calibration_file):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        print(images_left)
        images_right = glob.glob(dataset_dir + '/right/*.png')
        left_result_dir = os.path.join(dataset_dir + "Rectified", "left")
        right_result_dir = os.path.join(dataset_dir + "Rectified", "right")
        images_left.sort()
        images_right.sort()

        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"

        mkdir_overwrite(left_result_dir)
        mkdir_overwrite(right_result_dir)

        H = np.fromfile(calibration_file, dtype=np.float32).reshape((3, 3))

        print("Using Homography from file, with values: ")
        print(H)

        H = np.linalg.inv(H)
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            # warp right image
            img_r = cv2.warpPerspective(img_r, H, img_r.shape[::-1],
                                        cv2.INTER_LINEAR +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            # save images
            cv2.imwrite(os.path.join(left_result_dir,
                                     os.path.basename(image_left)), img_l)
            cv2.imwrite(os.path.join(right_result_dir,
                                     os.path.basename(image_right)), img_r)
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

aruco_dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
board = aruco.CharucoBoard_create(
        22, 16,
        2,
        1.5,
        aruco_dictionary)

print(board.chessboardCorners)
print(len(board.chessboardCorners))
print("Nearest Markers")
print(board.nearestMarkerCorners[0])
print(len(board.nearestMarkerCorners))
print("Nearest Markers id")
print(board.nearestMarkerIdx[0])
print(len(board.nearestMarkerIdx))

def calibrate_fisheye(self, allCorners, allIds, imsize):
    one_pts = board.chessboardCorners
    obj_points = []
    for i in range(len(allIds)):
        obj_pts_sub = []
        for j in range(len(allIds[i])):
            obj_pts_sub.append(one_pts[allIds[i][j]])
        obj_points.append(np.array(obj_pts_sub, dtype=np.float32))\

    term_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    return cv2.fisheye.calibrate(obj_points, allCorners, imsize, criteria = term_criteria)
    


def calibrate_stereo(self, allCorners_l, allIds_l, allCorners_r, allIds_r, imsize, cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r):
    left_corners_sampled = []
    right_corners_sampled = []
    obj_pts = []
    one_pts = self.board.chessboardCorners
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

        obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
        left_corners_sampled.append(
            np.array(left_sub_corners, dtype=np.float32))
        right_corners_sampled.append(
            np.array(right_sub_corners, dtype=np.float32))

    
    stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                            cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    if self.cameraModel == 'perspective':
        flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS # TODO(sACHIN): Try without intrinsic guess
        flags |= cv2.CALIB_RATIONAL_MODEL

        return cv2.stereoCalibrate(
            obj_pts, left_corners_sampled, right_corners_sampled,
            cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, imsize,
            criteria=stereocalib_criteria, flags=flags)
    elif self.fisheye.cameraModel == 'fisheye':
        flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS # TODO(sACHIN): Try without intrinsic guess
        return cv2.stereoCalibrate(
            obj_pts, left_corners_sampled, right_corners_sampled,
            cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, imsize,
            criteria=stereocalib_criteria, flags=flags)
        
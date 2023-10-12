import numpy as np
import depthai as dai
import cv2
from typing import *

def _get_mesh(mapX: np.ndarray, mapY: np.ndarray):
    mesh_cell_size = 16
    mesh0 = []
    # Creates subsampled mesh which will be loaded on to device to undistort the image
    for y in range(mapX.shape[0] + 1):  # iterating over height of the image
        if y % mesh_cell_size == 0:
            row_left = []
            for x in range(mapX.shape[1] + 1):  # iterating over width of the image
                if x % mesh_cell_size == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        row_left.append(mapY[y - 1, x - 1])
                        row_left.append(mapX[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        row_left.append(mapY[y - 1, x])
                        row_left.append(mapX[y - 1, x])
                    elif x == mapX.shape[1]:
                        row_left.append(mapY[y, x - 1])
                        row_left.append(mapX[y, x - 1])
                    else:
                        row_left.append(mapY[y, x])
                        row_left.append(mapX[y, x])
            if (mapX.shape[1] % mesh_cell_size) % 2 != 0:
                row_left.append(0)
                row_left.append(0)

            mesh0.append(row_left)

    mesh0 = np.array(mesh0)
    # mesh = list(map(tuple, mesh0))
    return mesh0

def get_mesh(calib_data: dai.CalibrationHandler, socket: dai.CameraBoardSocket, frame_size: Tuple[int,int], alpha):
    M1 = np.array(calib_data.getCameraIntrinsics(socket, frame_size))
    d1 = np.array(calib_data.getDistortionCoefficients(socket))
    R1 = np.identity(3)
    if alpha is not None:
        M2, _ = cv2.getOptimalNewCameraMatrix(M1, d1, frame_size, alpha, frame_size, True)
    else:
        M2 = M1
    mapX, mapY = cv2.initUndistortRectifyMap(M1, d1, R1, M2, frame_size, cv2.CV_32FC1)

    meshCellSize = 16
    mesh0 = []
    for y in range(mapX.shape[0] + 1):  # iterating over height of the image
        if y % meshCellSize == 0:
            rowLeft = []
            for x in range(mapX.shape[1]):  # iterating over width of the image
                if x % meshCellSize == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        rowLeft.append(mapX[y - 1, x - 1])
                        rowLeft.append(mapY[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        rowLeft.append(mapX[y - 1, x])
                        rowLeft.append(mapY[y - 1, x])
                    elif x == mapX.shape[1]:
                        rowLeft.append(mapX[y, x - 1])
                        rowLeft.append(mapY[y, x - 1])
                    else:
                        rowLeft.append(mapX[y, x])
                        rowLeft.append(mapY[y, x])
            if (mapX.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
            mesh0.append(rowLeft)

    mesh0 = np.array(mesh0)
    mesh_width = mesh0.shape[1] // 2
    mesh_height = mesh0.shape[0]
    mesh0.resize(mesh_width * mesh_height, 2)
    mesh = list(map(tuple, mesh0))
    mesh_dimensions = [mesh_width, mesh_height]
    return mesh, mesh_dimensions, M2
from depthai_calibration.calibration_utils import distance
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from scipy.stats import norm
import scipy as sp
import glob
from math import pi
import depthai as dai
import numpy as np
import cv2
from pathlib import Path
import matplotlib.colors as colors
import argparse
import os
import json

cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
          (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
          (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1
'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
          (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
          (1.0, 0.0, 0.0)),  # no green at 1
'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
          (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
          (1.0, 0.0, 0.0))   # no blue at 1
}
threshold = {"left": 0.75, "right": 0.75, "tof": 0.65, "rgb": 1.5, "color": 1.5, "vertical": 1.0}

def plot_reporjection(ax, display_corners, key, all_error, binary, width, height):
    center_x, center_y = width / 2, height / 2
    distances = [distance((center_x, center_y), point) for point in np.array(display_corners)]
    max_distance = max(distances)
    circle = plt.Circle((center_x, center_y), max_distance, color='black', fill=True, label = "Calibrated area", alpha = 0.2)
    ax.add_artist(circle)
    ax.set_title(f"Reprojection map camera {key}, {binary}")
    img = ax.scatter(np.array(display_corners).T[0], np.array(display_corners).T[1], c=all_error, cmap = GnRd, label = "Reprojected", vmin=0, vmax=threshold[key], s=7)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Reprojection error")
    ax.set_xlabel('Width')
    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    #ax.legend()
    ax.set_ylabel('Height')
    ax.grid()
    plt.tight_layout()
    return np.mean(all_error)

def plot_histogram(ax, key, error, binary, plt_gauss = False):
    ax.hist(error, range = [0,threshold[key]], bins = 100, edgecolor="Black", density = True)
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    x = np.linspace(xmin, xmax, len(error))
    if plt_gauss:
        mu, std = norm.fit(error)
        p = norm.pdf(x, mu, std)

        ax.plot(x, p, 'k', linewidth=2, label = "Fit Gauss: {:.4f} and {:.4f}".format(mu, std))
    else:
        mu, std = norm.fit(error) 
    param=sp.stats.lognorm.fit(error)
    pdf_fitted = sp.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2]) # fitted distribution
    shape, loc, scale = param
    ax.plot(x,pdf_fitted,'r-', label = r"Fit Log-Gauss: $\sigma=${:.4f} and $\mu$={:.4f}".format(shape, loc))
    ax.set_title(f"{key}, {binary}")
    ax.legend()
    ax.set_xlabel("Reprojection error[px]")
    ax.grid()
    return mu, std, param

def plot_contur(ax,fig,  calib, socket, imsize = (1280,800), index = 0):
    dist = np.array(calib.getDistortionCoefficients(socket))
    K = np.array(calib.getCameraIntrinsics(socket, resizeWidth=imsize[0], resizeHeight=imsize[1]))
    x,y = np.meshgrid(np.arange(imsize[0]),np.arange(imsize[1]))
             
    impoints = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    impoints = np.expand_dims(impoints.astype(float),1)
    impoints_undist = cv2.undistortPoints(impoints, K, dist, P=K)
    dist_diff = np.linalg.norm(impoints-impoints_undist, axis = 2)
    z = dist_diff.reshape(imsize[1],imsize[0])
    Con= ax.contourf(x, y, z, levels = 50, cmap=GnRd)
    fig.colorbar(Con, label='Distortion Difference')
    CS = ax.contour(z,levels = 50, alpha = 0.5, cmap = "RdYlGn")
    #ax.plot([],[],color = "White", label = f"min: {np.min(z)}, max: {np.max(z)}, mean: {np.mean(z)}")
    ax.set_xlabel("Width")
    ax.legend()
    #ax.clabel(CS, inline=0.5, fontsize=8)
    ax.set_ylabel("Height")

GnRd = colors.LinearSegmentedColormap('GnRd', cdict)

def plot_all(main, device, title, calib, save, folder = str(pathlib.Path(__file__).resolve().parent), display = False, binaries = None):
    if device == "OAK-D-SR":
        sockets = [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]
    else:
        sockets = [dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]
    if len(sockets) == 3:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
        fig.suptitle(title)
        ax1,ax1_c, ax2, ax2_c, ax3, ax3_c = axes.flatten()
        ax_ = [ax1, ax2, ax3]
        fig_hist, axes_hist = plt.subplots(nrows=3, ncols=1, figsize=(9, 14))
        fig_hist.suptitle(title)
        ax1_h, ax2_h, ax3_h = axes_hist.flatten()
        ax_hist = [ax1_h, ax2_h, ax3_h]
        ax_contur = [ax1_c, ax2_c, ax3_c]
    if len(sockets) == 2:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
        fig.suptitle(title)
        ax1,ax1_c, ax2, ax2_c = axes.flatten()
        ax_ = [ax1, ax2]
        fig_hist, axes_hist = plt.subplots(nrows=2, ncols=1, figsize=(9, 14))
        fig_hist.suptitle(title)
        ax1_h, ax2_h = axes_hist.flatten()
        ax_hist = [ax1_h, ax2_h]
        ax_contur = [ax1_c, ax2_c]
    index = 0
    mean = {}
    standard = {}
    params= {}
    reprojection_dict = {}
    for index, key in enumerate(main.stereo_calib.all_features.keys()):
        ax = ax_[index]
        ah = ax_hist[index]
        ac = ax_contur[index]
        binary = binaries[key]
        display_corners = main.stereo_calib.all_features[key]
        all_error = main.stereo_calib.all_errors[key]
        height = main.stereo_calib.height[key]
        width = main.stereo_calib.width[key]
        plot_contur(ac, fig, calib, sockets[index], imsize = (width,height), index = index)
        reprojection = plot_reporjection(ax, display_corners, key, all_error, binary, main.stereo_calib.width[key], main.stereo_calib.height[key])
        mu, std, param = plot_histogram(ah, key, all_error, binary)
        mean[key] = mu
        standard[key] = std
        params[key] = param
        reprojection_dict[key] = reprojection
    if len(sockets) == 3:
        fig.subplots_adjust(top=0.898,bottom=0.082, left=0.169, right=0.946, hspace=0.56, wspace=0.2)
        fig_hist.subplots_adjust(top=0.898, bottom=0.082, left=0.053, right=0.973, hspace=0.56, wspace=0.2)
    if len(sockets) == 2:
        fig.subplots_adjust(top=0.916,bottom=0.069,left=0.074,right=0.964,hspace=0.245,wspace=0.207)
        fig_hist.subplots_adjust(top=0.916,bottom=0.069,left=0.077,right=0.973,hspace=0.245,wspace=0.)
    if display:
        plt.show()
    if save:
        fig.savefig(folder + f'/images/rep_{title}_{binary}.png')
        fig_hist.savefig(folder + f'/images/hist_{title}_{binary}.png')
        plt.close(fig)
        plt.close(fig_hist)
    return reprojection_dict, mean, standard, params

import shutil
import os
from dataclasses import dataclass
@dataclass
class RectificationMaps:
    map_x: np.ndarray
    map_y: np.ndarray
def copy_session_folder(initial_path, final_path):
    # Define the source and destination paths for the 'capture' folder
    source_capture_path = os.path.join(initial_path, 'capture')
    destination_capture_path = os.path.join(final_path)
    
    # Check if the source capture path exists
    if not os.path.exists(source_capture_path):
        raise FileNotFoundError(f"The source path '{source_capture_path}' does not exist.")
    
    # Check if the destination path exists; if not, create it
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    
    # Delete all files in the destination_capture_path if it exists
    if os.path.exists(destination_capture_path):
        for root, dirs, files in os.walk(destination_capture_path):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    
    # Copy the content of the capture path to the destination path
    try:
        shutil.copytree(source_capture_path, destination_capture_path, dirs_exist_ok=True)
        print(f"Successfully copied '{source_capture_path}' to '{destination_capture_path}'")
    except Exception as e:
        raise RuntimeError(f"An error occurred while copying the 'capture' folder: {e}")
    for folder in os.listdir(destination_capture_path):
        if folder == "color":
            os.rename(destination_capture_path + "/color", destination_capture_path+ "/rgb")
    # Define the source and destination paths for the depth_test images
    left_source_path = os.path.join(initial_path, 'depth_test',"left;right", 'left')
    right_source_path = os.path.join(initial_path, 'depth_test',"left;right", 'right')
    
    left_destination_path = os.path.join(final_path, 'left.png')
    right_destination_path = os.path.join(final_path, 'right.png')

    # Check and copy the left image
    if os.path.exists(left_source_path) and os.listdir(left_source_path):
        left_image = os.listdir(left_source_path)[0]  # Assuming there's only one file in the folder
        left_image_source = os.path.join(left_source_path, left_image)
        try:
            shutil.copyfile(left_image_source, left_destination_path)
            print(f"Successfully copied '{left_image_source}' to '{left_destination_path}'")
        except Exception as e:
            raise RuntimeError(f"An error occurred while copying the left image: {e}")
    else:
        print(f"No files found in '{left_source_path}'")

    # Check and copy the right image
    if os.path.exists(right_source_path) and os.listdir(right_source_path):
        right_image = os.listdir(right_source_path)[0]  # Assuming there's only one file in the folder
        right_image_source = os.path.join(right_source_path, right_image)
        try:
            shutil.copyfile(right_image_source, right_destination_path)
            print(f"Successfully copied '{right_image_source}' to '{right_destination_path}'")
        except Exception as e:
            raise RuntimeError(f"An error occurred while copying the right image: {e}")
    else:
        print(f"No files found in '{right_source_path}'")


meshCellSize = 16

def rotate_mesh_90_ccw(map_x, map_y):
    direction = 1
    map_x_rot = np.rot90(map_x, direction)
    map_y_rot = np.rot90(map_y, direction)
    return map_x_rot, map_y_rot

def rotate_mesh_90_cw(map_x, map_y):
    direction = -1
    map_x_rot = np.rot90(map_x, direction)
    map_y_rot = np.rot90(map_y, direction)
    return map_x_rot, map_y_rot

def downSampleMesh(mapXL, mapYL, mapXR, mapYR):
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight


def create_mesh_on_host(calibData, leftSocket, rightSocket, resolution, vertical=False):
    width = resolution[0]
    height = resolution[1]

    M1 = np.array(calibData.getCameraIntrinsics(leftSocket, width, height))
    d1 = np.array(calibData.getDistortionCoefficients(leftSocket))
    M2 = np.array(calibData.getCameraIntrinsics(rightSocket, width, height))
    d2 = np.array(calibData.getDistortionCoefficients(rightSocket))

    T = np.array(calibData.getCameraTranslationVector(leftSocket, rightSocket, False))
    extrinsics = np.array(calibData.getCameraExtrinsics(leftSocket, rightSocket))
    extrinsics = extrinsics.flatten()
    R = np.array([
        [extrinsics[0], extrinsics[1], extrinsics[2]],
        [extrinsics[4], extrinsics[5], extrinsics[6]],
        [extrinsics[8], extrinsics[9], extrinsics[10]]
    ])

    T2 = np.array(calibData.getCameraTranslationVector(leftSocket, rightSocket, True))

    def calc_fov_D_H_V(f, w, h):
        return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.fisheye.stereoRectify(M1, d1, M2, d2, resolution, R, T)
    TARGET_MATRIX = M2
    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, TARGET_MATRIX, resolution, cv2.CV_32FC1)
    mapXV, mapYV = cv2.initUndistortRectifyMap(M2, d2, R2, TARGET_MATRIX, resolution, cv2.CV_32FC1)
    if vertical:
        baseline = abs(T2[1])*10
        focal = TARGET_MATRIX[0][0]
        mapXL_rot, mapYL_rot = rotate_mesh_90_ccw(mapXL, mapYL)
        mapXV_rot, mapYV_rot = rotate_mesh_90_ccw(mapXV, mapYV)
    else:
        baseline = abs(T2[0])*10
        focal = TARGET_MATRIX[1][1]
        mapXL_rot, mapYL_rot = mapXL, mapYL
        mapXV_rot, mapYV_rot = mapXV, mapYV
    leftMeshRot, verticalMeshRot = downSampleMesh(mapXL_rot, mapYL_rot, mapXV_rot, mapYV_rot)

    meshLeft = list(leftMeshRot.tobytes())
    meshVertical = list(verticalMeshRot.tobytes())
    focalScaleFactor = baseline * focal * 32
    print("Focal scale factor", focalScaleFactor)

    leftMap = RectificationMaps(map_x=mapXL, map_y=mapYL)
    verticalMap = RectificationMaps(map_x=mapXV, map_y=mapYV)

    return leftMap, verticalMap, meshLeft, meshVertical, focalScaleFactor
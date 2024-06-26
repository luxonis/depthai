from calibrate import Main
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
# Create the colormap using the dictionary
GnRd = colors.LinearSegmentedColormap('GnRd', cdict)
threshold = {"left": 1, "right": 1, "rgb": 1.5}

def rail_steps(steps: int) -> float:
    """
        Convert rail steps to mm
    """
    return steps / ((400) / (18 * pi))

def plot_reporjection(ax, display_corners, key, all_error, save):
    center_x, center_y = main.stereo_calib.width[key] / 2, main.stereo_calib.height[key] / 2
    distances = [distance((center_x, center_y), point) for point in np.array(display_corners)]
    max_distance = max(distances)
    circle = plt.Circle((center_x, center_y), max_distance, color='black', fill=True, label = "Calibrated area", alpha = 0.2)
    ax.add_artist(circle)
    ax.set_title(f"Reprojection map camera {key}")
    img = ax.scatter(np.array(display_corners).T[0], np.array(display_corners).T[1], c=all_error, cmap = GnRd, label = "Reprojected", vmin=0, vmax=threshold[key], s=3)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Reprojection error")
    ax.set_xlabel('Width')
    ax.set_xlim([0,main.stereo_calib.width[key]])
    ax.set_ylim([0,main.stereo_calib.height[key]])
    #ax.legend()
    ax.set_ylabel('Height')
    ax.grid()
    plt.tight_layout()
    return np.mean(all_error)

def plot_histogram(ax, key, error, save, plt_gauss = False):
    ax.hist(error, range = [0,threshold[key]], bins = 100, edgecolor="Black", density = True)
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    x = np.linspace(xmin, xmax, len(error))
    if plt_gauss:
        mu, std = norm.fit(error)
        p = norm.pdf(x, mu, std)

        ax.plot(x, p, 'k', linewidth=2, label = "Fit Gauss: {:.4f} and {:.4f}".format(mu, std))
    else:
        mu, std = 0, 0 
    param=sp.stats.lognorm.fit(error)
    pdf_fitted = sp.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2]) # fitted distribution
    shape, loc, scale = param
    ax.plot(x,pdf_fitted,'r-', label = r"Fit Log-Gauss: $\sigma=${:.4f} and $\mu$={:.4f}".format(shape, loc))
    ax.set_title(key)
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

def depth_evaluation(calib, left_array, right_array, depth_on_charucos, title, folder = str(pathlib.Path(__file__).resolve().parent), display = False):
    device = dai.Device()
    pipeline = dai.Pipeline()


    ### NEED TO ADD JSON CALIBRATION YOU WANT ###
    calib = dai.CalibrationHandler(calib)
    pipeline.setCalibrationData(calib)

    left_socket = dai.CameraBoardSocket.LEFT
    right_socket = dai.CameraBoardSocket.RIGHT
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoLeft.setBoardSocket(left_socket)
    monoRight.setBoardSocket(right_socket)
    monoLeft.setFps(5)
    monoRight.setFps(5)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

    xin_left = pipeline.create(dai.node.XLinkIn)
    xin_left.setStreamName("left")
    xin_right = pipeline.create(dai.node.XLinkIn)
    xin_right.setStreamName("right")

    xin_left_out = xin_left.out
    xin_right_out = xin_right.out

    xoutDisparity = pipeline.create(dai.node.XLinkOut)
    xoutDisparity.setStreamName("depth")
    stereo = pipeline.create(dai.node.StereoDepth)

    #stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.enableDistortionCorrection(True)
    xin_left_out.link(stereo.left)
    xin_right_out.link(stereo.right)
    stereo.depth.link(xoutDisparity.input)

    def send_images(frames):
        input_queues = {"left": device.getInputQueue("left"), "right": device.getInputQueue("right")}
        ts = dai.Clock.now()
        for name, image in dummie_image.items():
            w, h = image.shape[1], image.shape[0]
            frame = cv2.resize(image, (w, h), cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if name == "left":
                number = int(1)
                if not rectification:
                    frame = cv2.remap(frame, leftMap.map_x, leftMap.map_y, cv2.INTER_LINEAR)
            if name == "right":
                number = int(2)
                if not rectification:
                    frame = cv2.remap(frame, rightMap.map_x, rightMap.map_y, cv2.INTER_LINEAR)
            img = dai.ImgFrame()
            img.setData(frame.reshape(h * w))
            img.setTimestamp(ts)
            img.setWidth(w)
            img.setHeight(h)
            img.setInstanceNum(number)
            #img.setType(dai.ImgFrame.Type.RAW8)
            input_queues[name].send(img)
    rect_display = True
    if rect_display:
        xoutRectifiedLeft = pipeline.create(dai.node.XLinkOut)
        xoutRectifiedLeft.setStreamName("rectified_left")
        stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    if rect_display:
        xoutRectifiedRight = pipeline.create(dai.node.XLinkOut)
        xoutRectifiedRight.setStreamName("rectified_right")
        stereo.rectifiedRight.link(xoutRectifiedRight.input)

    rectification = True
    if not rectification:
        #### Make your own rectification as you wish ###
        meshLeft = None
        meshRight = None
        leftMap = None
        rightMap = None
        stereo.loadMeshData(meshLeft, meshRight)
        stereo.setRectification(False)

    output_queues = [xoutDisparity.getStreamName(), "rectified_left", "rectified_right"]

    device.startPipeline(pipeline)

    ### NEED TO ADD IMAGES YOU WANT, HERE IS JUST DUMMY FRAME ###
    destroy = False
    index = 0
    if depth_on_charucos:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(17, 8))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        _ax_h = [ax1, ax3, ax5]
        _ax_m = [ax2, ax4, ax6]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2 , figsize=(17, 8))
        ax1, ax2 = axes.flatten()
        _ax_h = [ax1]
        _ax_m = [ax2]
    fig.subplots_adjust(top=0.87,bottom=0.104,left=0.037,right=0.961,hspace=0.2,wspace=0.102)
    fig.suptitle(title)
    while True:
        queues = {name: device.getOutputQueue(name, 4, False) for name in output_queues}
        while True:
            GT = rail_steps(float(left_array[index].split("_")[len(left_array[index].split("_")) - 1].split(".")[0])) - 30
            GT = 75
            dummie_image = {"left": cv2.imread(left_array[index]), "right": cv2.imread(right_array[index])}
            send_images(dummie_image)
            index += 1
            frames = {name: queue.get().getCvFrame() for name, queue in queues.items()}
            for name, frame in frames.items():
                cv2.imshow(name, frame)
                key = cv2.waitKey(1)
                if name == "depth":
                    ax = _ax_h[index-1]
                    ax.set_title(left_array[index-1])
                    error = frame.flatten()/1000
                    range_min = GT/100 - 0.01
                    range_max = GT/100 + 0.08
                    error = error[(error >= range_min) & (error <= range_max)]
                    num_filtered_points = len(error)/len(frame.flatten())
                    bins = 100
                    bin_width = (range_max - range_min) / bins
                    hist_data, bin_edges, _ = ax.hist(error, bins=bins, range=[range_min, range_max], edgecolor="black", density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    mu, std = norm.fit(error)
                    p = norm.pdf(bin_centers, mu, std)
                    ax.plot(bin_centers, p, color = "Black", linewidth=2, label="Fit Gauss: μ={:.4f}, σ={:.4f}".format(mu, std))

                    ax.set_xlabel("Distance[m]")
                    ax.legend()
                    ax.grid()
                    ax_m =_ax_m[index-1]
                    image = ax_m.imshow(frame/1000, vmin = GT/100 - 0.01, vmax = GT/100 + 0.06)
                    fig.colorbar(image, label = "Distance [m]", ax=ax_m)
                    if index == 3 and depth_on_charucos:
                        destroy = True
                    elif index == 1 and not depth_on_charucos:
                        destroy = True
                if key == ord("q"):
                    destroy = True
                    break
            if destroy:
                break
        if destroy:
            break
    device.close()
    fig.savefig(folder + f'/depth_{title}.png')
    if display:
        plt.show()
    plt.close(fig)
    return mu, std, num_filtered_points

def plot_all(title, calib, save, folder = str(pathlib.Path(__file__).resolve().parent), display = False):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
    fig.suptitle(title)
    ax1,ax1_c, ax2, ax2_c, ax3, ax3_c = axes.flatten()
    ax_ = [ax1, ax2, ax3]
    fig_hist, axes_hist = plt.subplots(nrows=3, ncols=1, figsize=(9, 14))
    fig_hist.suptitle(title)
    ax1_h, ax2_h, ax3_h = axes_hist.flatten()
    ax_hist = [ax1_h, ax2_h, ax3_h]
    ax_contur = [ax1_c, ax2_c, ax3_c]
    index = 0
    mean = {}
    standard = {}
    params= {}
    reprojection_dict = {}
    sockets = [dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]
    for index, key in enumerate(main.stereo_calib.all_features.keys()):
        ax = ax_[index]
        ah = ax_hist[index]
        ac = ax_contur[index]
        display_corners = main.stereo_calib.all_features[key]
        all_error = main.stereo_calib.all_errors[key]
        height = main.stereo_calib.height[key]
        width = main.stereo_calib.width[key]
        plot_contur(ac, fig, calib, sockets[index], imsize = (width,height), index = index)
        reprojection = plot_reporjection(ax, display_corners, key, all_error, save)
        mu, std, param = plot_histogram(ah, key, all_error, save)
        mean[key] = mu
        standard[key] = std
        params[key] = param
        reprojection_dict[key] = reprojection

    fig.subplots_adjust(top=0.898,bottom=0.082, left=0.169, right=0.946, hspace=0.56, wspace=0.2)
    fig_hist.subplots_adjust(top=0.898, bottom=0.082, left=0.053, right=0.973, hspace=0.56, wspace=0.2)
    if True:
        plt.show()
    if save:
        fig.savefig(folder + f'/rep_{title}.png')
        fig_hist.savefig(folder + f'/hist_{title}.png')
        plt.close(fig)
        plt.close(fig_hist)
    return reprojection_dict, mean, standard, params

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


GnRd = colors.LinearSegmentedColormap('GnRd', cdict)
device = "OAK-D-PRO"
save_folder = str(pathlib.Path(__file__).resolve().parent)

static = ['-s', '5.0', '-nx', '29', '-ny', '29', '-ms', '3.7', '-dbg', '-m', 'process', '-brd', device + ".json", "-scp", save_folder]

def generate_binary_strings(n):
    binary_strings = []
    base_string = "000000001"

    for i in range(n):
        # Generate the next binary string by converting i to binary and adding leading zeros
        binary_value = bin(i)[2:].zfill(9)
        binary_string = list(base_string)

        # Fill the generated binary value into the base string starting from the end
        for j in range(9):
            if binary_value[j] == '1':
                binary_string[j] = '1'
        if "".join(binary_string) not in binary_strings:
            binary_strings.append("".join(binary_string))
    return binary_strings

# Generate the first 10 binary strings with the specified format
calibration_models = generate_binary_strings(150)
print(calibration_models)

display_graphs = False
save = True
depth_on_charucos = False
mean = {}
standard = {}
params_0 = {}
params_2 = {}
reprojection_aray = {}
depth_mean = []
depth_standard = []
depth_fill = []
for k, binary in enumerate(calibration_models):
    print(f"ON {k}/{len(calibration_models)}")
    dynamic = static + ['-pccm', 'left=' + binary, 'right=' + binary, "rgb=" + binary]
    main = Main(dynamic)
    main.run()
    calib = dai.CalibrationHandler(main.calib_dest_path)
    reprojection, mu, std, param = plot_all(binary + " " + device, calib, save, display = display_graphs)
    for key in reprojection.keys():
        shape, loc, scale = param[key]
        if key not in mean.keys():
            mean[key] = []
        mean[key].append(mu[key])
        if key not in standard.keys():
            standard[key] = []
        standard[key].append(std[key])
        if key not in params_0.keys():
            params_0[key] = []
        params_0[key].append(loc)
        if key not in params_2.keys():
            params_2[key] = []
        params_2[key].append(shape)
        if key not in reprojection_aray.keys():
            reprojection_aray[key] = []
        reprojection_aray[key].append(reprojection[key])


    calib = main.calib_dest_path
    left_array = []
    right_array = []
    if depth_on_charucos:
        filepath = main.dataset_path
        left_path = filepath + '/' + "left"
        left_files = glob.glob(left_path + "/*")
        left_array = []
        right_array = []
        for file in left_files:
            if float(file.split("_")[2]) == 0.0 and float(file.split("_")[3]) == 0.0:
                right_array.append(file.replace("left", "right"))
                left_array.append(file)
    else:
        left_array.append("D:/FAKS, FMF/Studentska dela/Luxonis/depthai/dataset/left_0_-180_0_1303.0.png")
        right_array.append("D:/FAKS, FMF/Studentska dela/Luxonis/depthai/dataset/right_0_-180_0_1303.0.png")
    mu, sigma, fillrate = depth_evaluation(calib, left_array, right_array, depth_on_charucos, binary, display = display_graphs)
    depth_mean.append(mu)
    depth_standard.append(sigma)
    depth_fill.append(fillrate)
# First plot



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(17, 8))
fig.subplots_adjust(top=0.922,
bottom=0.145,
left=0.065,
right=0.986,
hspace=0.03,
wspace=0.2)
fig.suptitle("Rectification error")

#ax1.scatter(x1, mean, marker="x", label="Mean of Gauss", color="Blue")
#ax1.scatter(x1, standard, label="Standard deviation of Gauss", color="Blue")
color = ["Green", "Red", "Blue"]
for index, key in enumerate(reprojection.keys()):
    x1 = np.linspace(0, len(params_0[key]) + 1, len(params_0[key]))
    ax1.scatter(x1, params_0[key], marker="x", label=f"Mean: {key}", color=color[index])
    ax1.scatter(x1, params_2[key], label=f"Std: {key}", color=color[index])
ax1.legend()
ax1.set_ylabel("Reprojection [px]")
ax1.grid()

# Second plot
x2 = np.linspace(0, len(depth_fill) + 1, len(depth_fill))
ax2.scatter(x2, depth_mean, label="Mean of depth", color="Red")
ax2.scatter(x2, depth_fill, marker="x", label="Fill rate", color="Red")
ax2.tick_params(axis='y', labelcolor='black', rotation=45, ha='right')
ax2.legend()
ax3 = ax2.twinx()
ax3.scatter(x2, depth_standard, label="Standard deviation", color="Blue")
ax2.set_ylabel("Error [m]/Fillrate [%]")
ax3.set_ylabel("Deviation [m]", color='blue')
ax2.grid()

plt.tight_layout()
plt.show()


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
    img = ax.scatter(np.array(display_corners).T[0], np.array(display_corners).T[1], c=all_error, cmap = GnRd, label = "Reprojected", vmin=0, vmax=threshold[key])
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

def plot_histogram(ax, key, error, save):
    ax.hist(error, range = [0,threshold[key]], bins = 100, edgecolor="Black", density = True)
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    x = np.linspace(xmin, xmax, len(error))
    mu, std = norm.fit(error)
    p = norm.pdf(x, mu, std)
    
    ax.plot(x, p, 'k', linewidth=2, label = "Fit Gauss: {:.4f} and {:.4f}".format(mu, std))
    param=sp.stats.lognorm.fit(error)
    pdf_fitted = sp.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2]) # fitted distribution
    ax.plot(x,pdf_fitted,'r-', label = "Fit Log-Gauss: {:.4f} and {:.4f}".format(param[2], param[0]))
    ax.set_title(key)
    ax.legend()
    ax.set_xlabel("Reprojection error[px]")
    ax.grid()
    return mu, std, param

def depth_evaluation(calib, left_array, right_array, depth_on_charucos):
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
            img.setType(dai.ImgFrame.Type.RAW8)
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
        fig, axes = plt.subplots(nrows=3, ncols=2)
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        _ax_h = ax1, ax3, ax5
        _ax_m = ax2, ax4, ax6
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axes.flatten()
        _ax_h = ax1
        _ax_m = ax2
    fig.suptitle(left_array[index])
    while True:
        queues = {name: device.getOutputQueue(name, 4, False) for name in output_queues}
        while True:
            GT = rail_steps(float(left_array[index].split("_")[len(left_array[index].split("_")) - 1].split(".")[0])) - 30
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
                    ax.hist(error, bins = 100, range=[GT/100 - 0.15,GT/100 + 0.15], edgecolor = "Black", density = True)
                    xmin, xmax = ax.set_xlim()
                    ymin, ymax = ax.set_ylim()
                    x = np.linspace(xmin, xmax, len(error))
                    mu, std = norm.fit(error)
                    p = norm.pdf(x, mu, std)
                    ax.plot(x, p, 'k', linewidth=2, label = "Fit Gauss: {:.4f} and {:.4f}".format(mu, std))
                    try:
                        param=sp.stats.lognorm.fit(error)
                        pdf_fitted = sp.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2]) # fitted distribution
                        ax.plot(x,pdf_fitted,'r-', label = "Fit Log-Gauss: {:.4f} and {:.4f}".format(param[2], param[0]))
                    except:
                        pass
                    ax.set_xlabel("Distance[m]")
                    ax.legend()
                    ax.grid()
                    ax_m =_ax_m[index-1]
                    image = ax_m.imshow(frame/1000, vmin = GT/100 - 0.15, vmax = GT/100 + 0.15)
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

def plot_all(title, save, folder = str(pathlib.Path(__file__).resolve().parent)):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 14))
    fig.suptitle(title)
    ax1, ax2, ax3 = axes.flatten()
    ax_ = [ax1, ax2, ax3]
    fig_hist, axes_hist = plt.subplots(nrows=3, ncols=1, figsize=(9, 14))
    fig_hist.suptitle(title)
    ax1_h, ax2_h, ax3_h = axes_hist.flatten()
    ax_hist = [ax1_h, ax2_h, ax3_h]
    index = 0
    for key in main.stereo_calib.all_features.keys():
        ax = ax_[index]
        ah = ax_hist[index]
        display_corners = main.stereo_calib.all_features[key]
        all_error = main.stereo_calib.all_errors[key]
        reprojection = plot_reporjection(ax, display_corners, key, all_error, save)
        mu, std, param = plot_histogram(ah, key, all_error, save)
        index += 1
    fig.subplots_adjust(top=0.898,bottom=0.082, left=0.169, right=0.946, hspace=0.56, wspace=0.2)
    fig_hist.subplots_adjust(top=0.898, bottom=0.082, left=0.053, right=0.973, hspace=0.56, wspace=0.2)
    if save:
        fig.savefig(folder + f'/{title}.png')
        fig_hist.savefig(folder + f'/{title}_hist.png')
    return reprojection, mu, std, param

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
calibration_models = [
    "000000001",
    "000000011",
    "000000101",
    "000000111",
    "000001001",
    "000001011",
    "000001101",
    "000001111",
    "000010001",
    "000010011"
]
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
        binary_strings.append("".join(binary_string))

    return binary_strings

# Generate the first 10 binary strings with the specified format
binary_strings = generate_binary_strings(3)
print(binary_strings)

display_graphs = False
save = True
depth_on_charucos = True
mean = []
standard = []
params_0 = []
params_2 = []
for binary in calibration_models:
    dynamic = static + ['-pccm', 'left=' + binary, 'right=' + binary, "rgb=" + binary]
    main = Main(dynamic)
    main.run()

    reprojection, mu, std, param = plot_all(device + " "  + binary, save)
    mean.append(mu)
    standard.append(std)
    params_0.append(param[0])
    params_2.append(param[2])
    plt.tight_layout()
    if display_graphs:
        plt.show()
    else:
        plt.close()

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
    #depth_evaluation(calib, left_array, right_array, depth_on_charucos)
    #if display_graphs:
    #    plt.show()
    #else:
    #    plt.close()
x = np.linspace(0,len(params_0), len(params_0))
fig, ax = plt.subplots()
ax.scatter(x, mean, label = "Mean of Gauss")
ax.scatter(x, standard, label = "Standard deviation of Gauss")
ax.scatter(x, params_0, label = "Mean of Log-Gauss")
ax.scatter(x, params_2, label = "Standard deviation of Log-Gauss")
ax.legend()
ax.set_ylabel("Reprojection [px]")
ax.grid()
ax.set_xticks(range(1, len(binary_strings)+1))
ax.set_xticklabels(binary_strings, rotation=45, ha='right')
plt.show()
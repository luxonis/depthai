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
import argparse
import os
import json
from utils_plots import *


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
threshold = {"left": 0.75, "right": 0.75, "tof": 0.65, "rgb": 1.5, "color": 1.5, "vertical": 1.0}

parser = argparse.ArgumentParser()
parser.add_argument('-folder', type=str, default=None, help="Folder to session on which to run calibration.")
parser.add_argument('-device', type=str, default=None, help="Name of the device, on which the calibration is run.")
parser.add_argument('-config', type=str, default=None, help="Define config file.")
parser.add_argument('-number', type=int, default=512, help="Number of run combinations.")
parser.add_argument('-gt', type=float, default=1.7, help="Define groung truth for the depth test in m.")
parser.add_argument('-min_max', type=float, default=0.15, help="Define the distribution plot of ground truth.")
parser.add_argument("-full", help="Run the full 512 models", default=False, action="store_true")
parser.add_argument("-save", help="Save all the graphs which are created", default=True, action="store_true")
parser.add_argument("-graphs", help="Display all the graphs", default=False, action="store_true")
parser.add_argument("-disable_depth", help="Disable the depth test from device", default=True, action="store_true")
parser.add_argument('-save_folder', type=str, default=None, help="Folder where all will be saved.")
parser.add_argument('-c_model', type=str, default=None, help="Use fisheye model instead perspective.")
parser.add_argument('-session', type=str, default=None, help="Folder to a session on which calibration is performed.")
args = parser.parse_args()

data_path = str(pathlib.Path(__file__).resolve().parent) + "/dataset"
if args.session is not None:
    copy_session_folder(args.session, data_path)
device = args.device
if args.save_folder is None:
    print(pathlib.Path(__file__).resolve().parent)
    save_folder = str(pathlib.Path(__file__).resolve().parent) + "/" + device
else:
    save_folder = args.save_folder
os.makedirs(save_folder+ "/jsons/", exist_ok=True)
os.makedirs(save_folder+ "/images/", exist_ok=True)
static = ['-s', '5.0', '-nx', '29', '-ny', '29', '-ms', '3.7', '-dbg',  '-m', 'process', '-brd', args.device + ".json", "-scp", save_folder + "/jsons/"]
if args.c_model is not None:
    static += ["-cm", "fisheye"]

if args.config is None:
    config_file = str(pathlib.Path(__file__).resolve().parent) + "/calibration_config.json"
else:
    config_file = args.config


# Generate the first 10 binary strings with the specified format
n = args.number

with open(save_folder + "/" + f"calib_search_{n}.csv", 'w') as file:
    file.write("Binary,Camera,Mean,Standard,Reprojection_mean,Gauss_mean,Gauss_standard,Depth_mean[m],Depth_standard[m],Fillrate[%]\n")
display_graphs = args.graphs
save = args.save
depth_on_charucos = False

def rail_steps(steps: int) -> float:
    """
        Convert rail steps to mm
    """
    return steps / ((400) / (18 * pi))

def depth_evaluation(main, calib, left_array, right_array, depth_on_charucos, title, folder = str(pathlib.Path(__file__).resolve().parent), display = False):
    device = dai.Device()
    pipeline = dai.Pipeline()

    calib = dai.CalibrationHandler(calib)
    pipeline.setCalibrationData(calib)
    left_id = calib.getStereoLeftCameraId()
    right_id = calib.getStereoRightCameraId()

    left_socket = dai.CameraBoardSocket.LEFT
    right_socket = dai.CameraBoardSocket.RIGHT
    if main.board_config['cameras'][left_id.name]["type"] == 'mono':
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    else:
        monoLeft = pipeline.create(dai.node.ColorCamera)
        monoRight = pipeline.create(dai.node.ColorCamera)
        monoLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        monoRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    if False:
        monoLeft.setIspScale(2, 3)
        monoRight.setIspScale(2, 3)
    monoLeft.setBoardSocket(left_socket)
    monoRight.setBoardSocket(right_socket)
    monoLeft.setFps(5)
    monoRight.setFps(5)

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
            #GT = rail_steps(float(left_array[index].split("_")[len(left_array[index].split("_")) - 1].split(".")[0])) - 30
            GT = 105
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
                    range_min = args.gt - args.min_max
                    range_max = args.gt + args.min_max
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
                    image = ax_m.imshow(frame/1000, vmin = range_min, vmax = range_max)
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
    fig.savefig(folder + f'/images/depth_{title}.png')
    if display:
        plt.show()
    plt.close(fig)
    return mu, std, num_filtered_points

def generate_binary_strings(n, model = "perspective"):
    binary_strings = []
    if model == "perspective":
        base_string = "000000001"
        size = 9
    else:
        base_string = "0001"
        size = 4
    for i in range(n):
        binary_value = bin(i)[2:].zfill(size)
        binary_string = list(base_string)

        for j in range(size):
            if binary_value[j] == '1':
                binary_string[j] = '1'
        if "".join(binary_string) not in binary_strings and "".join(binary_string) not in ["111111000", "111111100", "111111110", "111111111", "111111010", "111111001"]:
            binary_strings.append("".join(binary_string))
    return binary_strings

if not args.full:
    if os.path.exists(config_file):
        with open(config_file, 'r') as infile:
            results = json.load(infile)
        if device in results.keys():
            calibration_models_dict = results[device]["binaries"]
        else:
            calibration_models = generate_binary_strings(n*2, args.c_model)
            calibration_models_dict = {'left': calibration_models, "right": calibration_models, "rgb": calibration_models}
    else:
        print("Config file does not exsists, defaulting back to the normal.")
        calibration_models = generate_binary_strings(n*2, args.c_model)
        calibration_models_dict = {'left': calibration_models, "right": calibration_models, "rgb": calibration_models}
else:
    calibration_models = generate_binary_strings(n*2, args.c_model)
    calibration_models_dict = {'left': calibration_models, "right": calibration_models, "rgb": calibration_models}

mean = {}
standard = {}
params_0 = {}
params_2 = {}
reprojection_aray = {}
depth_mean = []
depth_standard = []
depth_fill = []
calibration_models = []
for k in range(len(calibration_models_dict["left"])):
    static = static + ["-pccm"]
    current_binaries = {}
    for key in calibration_models_dict:
        binary = calibration_models_dict[key][k]
        static += [key + "=" + binary]
        current_binaries[key] = binary
    dynamic = static
    calibration_models.append(binary)
    print(f"ON {k}/{len(calibration_models_dict['left'])}")
    main = Main(dynamic)
    main.run()
    calib = dai.CalibrationHandler(main.calib_dest_path)
    reprojection, mu, std, param = plot_all(main, device, device, calib, save, folder = save_folder, display = display_graphs, binaries = current_binaries)
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
        with open(save_folder + "/" + f"calib_search_{n}.csv", 'a') as file:
            file.write(f"{current_binaries[key]},{key},{mu[key]},{std[key]},{reprojection[key]},{loc},{shape},{0},{0},{0}\n")


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
        left_array.append(data_path +"/left.png")
        right_array.append(data_path + "/right.png")
    if args.disable_depth:
        mu, sigma, fillrate = depth_evaluation(main, calib, left_array, right_array, depth_on_charucos, binary, folder = save_folder, display = display_graphs)
        with open(save_folder + "/" + f"calib_search_{n}.csv", 'a') as file:
            file.write(f"{binary},stereo,{0},{0},{0},{0},{0},{mu},{sigma},{fillrate}\n")
    else:
        mu, sigma, fillrate = 0, 0, 0
    depth_mean.append(mu)
    depth_standard.append(sigma)
    depth_fill.append(fillrate)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(17, 8))
fig.subplots_adjust(top=0.922,
bottom=0.145,
left=0.065,
right=0.986,
hspace=0.03,
wspace=0.2)
fig.suptitle("Rectification error")

color = ["Green", "Red", "Blue", "Yellow"]
for index, key in enumerate(reprojection.keys()):
    x1 = np.linspace(0, len(params_0[key]) + 1, len(params_0[key]))
    ax1.scatter(x1, mean[key], marker="x", label=f"Mean of Gauss, {key}", color=color[index])
    ax1.scatter(x1, standard[key], label=f"Standard deviation of Gauss {key}", color=color[index])
ax1.legend()
ax1.set_ylabel("Reprojection [px]")
ax1.grid()

# Second plot
x2 = np.linspace(0, len(depth_fill) + 1, len(depth_fill))
ax2.scatter(x2, depth_mean, label="Mean of depth", color="Red")
ax2.scatter(x2, depth_fill, marker="x", label="Fill rate", color="Red")
ax2.tick_params(axis='y', labelcolor='black', rotation=45)
ax2.legend()
ax3 = ax2.twinx()
ax3.scatter(x2, depth_standard, label="Standard deviation", color="Blue")
ax2.set_ylabel("Error [m]/Fillrate [%]")
ax3.set_ylabel("Deviation [m]", color='blue')
ax2.set_xticks(x2)  # Set x-ticks to the positions of x2
ax2.set_xticklabels(calibration_models, rotation=45)
ax2.grid()

plt.tight_layout()
plt.show()

overall_all = {}
best_models = {}
best_results = {}
for index, key in enumerate(reprojection.keys()):
    reprojection_mean_evaluated = []
    reprojection_std_evaluated = []
    overall = []
    for i in range(len(calibration_models)):
        depth_mean_evaluated = np.abs(depth_mean[i] - depth_mean[0]) / depth_mean[0]
        depth_standard_evaluated = (depth_standard[i] - depth_standard[0]) / depth_standard[0]
        reprojection_mean_evaluated = (mean[key][i]- mean[key][0]) / mean[key][0]
        reprojection_std_evaluated = (standard[key][i]- standard[key][0]) / standard[key][0]
        overall.append((reprojection_mean_evaluated + reprojection_std_evaluated)*100)

    paired = list(zip(calibration_models, overall))
    sorted_pairs = sorted(paired, key=lambda x: x[1])
    lowest_half = sorted_pairs[:len(sorted_pairs)//2]

    best_models[key] = [x[0] for x in lowest_half]
    best_results[key] = [x[1] for x in lowest_half]
    overall_all[key] = overall

if os.path.exists(config_file):
    with open(config_file, 'r') as infile:
        results = json.load(infile)
    if device in results.keys():
        results[device]["binaries"] = best_models
        results[device]["MXID"] = [device]
    else:
        results = {}
        results[device] = {}
        results[device]["binaries"] = best_models
        #FIX THIS SO THE MXID is read
        results[device]["MXID"] = [device]
else:
    results = {}
    results[device] = {}
    results[device]["binaries"] = best_models
    #FIX THIS SO THE MXID is read
    results[device]["MXID"] = [device]
with open(config_file, 'w') as outfile:
    json.dump(results, outfile)

print("Done with everything.")
x2 = np.linspace(0, len(overall) + 1, len(overall))
fig, (ax) = plt.subplots(1, 1, sharex=True, figsize=(17, 8))
fig.suptitle(f"Overall improvement of reprojection + depth of {device}")
for index, key in enumerate(overall_all.keys()):
    x2 = np.linspace(0, len(overall_all[key]) + 1, len(overall_all[key]))
    ax.scatter(x2, overall_all[key], label = key)
ax.legend()
ax.grid()
ax.set_xticks(x2)  # Set x-ticks to the positions of x2
ax.set_xticklabels(calibration_models, rotation=45)
ax.set_ylabel("Overall improvement[%]")
plt.show()
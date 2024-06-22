from calibrate import Main
from depthai_calibration.calibration_utils import distance
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from scipy.stats import norm
import scipy as sp

threshold = {"left": 1, "right": 1, "rgb": 1.5}
def plot_reporjection(ax, display_corners, key, all_error):
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
    return np.mean(all_error)

def plot_histogram(ax, key, error):
    ax.hist(error, range = [0,threshold[key]], bins = 100, edgecolor="Black", density = True)
    xmin, xmax = ax.set_xlim()
    ymin, ymax = ax.set_ylim()
    x = np.linspace(xmin, xmax, len(error))
    mu, std = norm.fit(error)
    p = norm.pdf(x, mu, std)
    
    ax.plot(x, p, 'k', linewidth=2, label = "Fit Gauss: {:.2f} and {:.2f}".format(mu, std))
    param=sp.stats.lognorm.fit(error)
    print(param)
    pdf_fitted = sp.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2]) # fitted distribution
    ax.plot(x,pdf_fitted,'r-', label = "Fit Log-Gauss: {:.2f} and {:.2f}".format(param[2], param[0]))
    ax.set_title(key)
    ax.legend()
    ax.set_xlabel("Reprojection error[px]")
    ax.grid()
    return mu, std

def plot_all():
    fig, axes = plt.subplots(nrows=3, ncols=1)
    ax1, ax2, ax3 = axes.flatten()
    ax_ = [ax1, ax2, ax3]

    fig_hist, axes_hist = plt.subplots(nrows=3, ncols=1)
    ax1_h, ax2_h, ax3_h = axes_hist.flatten()
    ax_hist = [ax1_h, ax2_h, ax3_h]
    index = 0
    for key in main.stereo_calib.all_features.keys():
        ax = ax_[index]
        ah = ax_hist[index]
        display_corners = main.stereo_calib.all_features[key]
        all_error = main.stereo_calib.all_errors[key]
        reprojection = plot_reporjection(ax, display_corners, key, all_error)
        mu, std = plot_histogram(ah, key, all_error)
        index += 1

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
left_binary = "000000000"
right_binary = "000000000"
color_binary = "000000000"


dynamic = static + ['-pccm', 'left=' + left_binary, 'right=' + right_binary, "rgb=" + color_binary]
main = Main(dynamic)
main.run()

plot_all()
plt.show()
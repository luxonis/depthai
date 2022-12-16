import cv2
import depthai

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import StereoColor

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', fps=5)
    stereo = oak.create_stereo('400p', fps=5)

    stereo.configure_postprocessing(
        colorize=StereoColor.RGB,
        colormap=cv2.COLORMAP_JET,
        wls_filter=True,
        wls_lambda=8000,
        wls_sigma=1.5
    )

    # Record RGB and disparity to records folder
    # Record doesn't work with visualize so the config is ignored
    oak.record([color.out.main, stereo.out.disparity], 'records')

    # Record depth only
    # oak.visualize(stereo.out.depth, record='depth.avi')

    oak.start(blocking=True)

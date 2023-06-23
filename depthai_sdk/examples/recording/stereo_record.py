import cv2

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import StereoColor

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', fps=30)
    stereo = oak.create_stereo('400p', fps=30)

    stereo.config_postprocessing(
        colorize=StereoColor.RGB,
        colormap=cv2.COLORMAP_JET
    )

    stereo.config_wls(
        wls_level='high'  # options: 'low', 'medium', 'high'
    )

    # Record RGB and disparity to records folder
    # Record doesn't work with visualize so the config is ignored
    # oak.record([color.out.main, stereo.out.disparity], 'records')

    # Record depth only
    oak.visualize(stereo.out.disparity, record_path='disparity.mp4')

    oak.start(blocking=True)

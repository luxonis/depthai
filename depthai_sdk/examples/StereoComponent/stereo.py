import cv2

from depthai_sdk import OakCamera
from depthai_sdk.components.stereo_component import WLSLevel
from depthai_sdk.visualize.configs import StereoColor

with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=30)
    stereo.config_postprocessing(
        colorize=StereoColor.GRAY,
        colormap=cv2.COLORMAP_BONE,
        wls_filter=True,
        wls_level=WLSLevel.HIGH
    )

    oak.visualize(stereo.out.disparity)
    oak.start(blocking=True)

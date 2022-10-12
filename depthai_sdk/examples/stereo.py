import depthai

from depthai_sdk import OakCamera

with OakCamera(usbSpeed=depthai.UsbSpeed.HIGH) as oak:
    stereo = oak.create_stereo('400p', fps=60, clickable=True)
    stereo.configure_postprocessing(
        colorize=True,
        wls_filter=True,
        wls_lambda=8000,
        wls_sigma=1.5
    )

    oak.visualize(stereo.out.depth)
    oak.start(blocking=True)

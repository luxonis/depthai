import depthai

from depthai_sdk import OakCamera

with OakCamera(usbSpeed=depthai.UsbSpeed.HIGH) as oak:
    stereo = oak.create_stereo('800p', fps=30)
    stereo.configure_postprocessing(
        colorize=False,
        wls_filter=True,
        wls_lambda=8000,
        wls_sigma=1.5
    )

    oak.visualize(stereo.out.depth)
    oak.start(blocking=True)

from depthai_sdk import OakCamera

with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=60)
    stereo.configure_postprocessing(
        colorize=True,
        wls_filter=True,
        wls_lambda=8000,
        wls_sigma=1.5
    )

    oak.visualize(stereo.out.disparity)
    oak.start(blocking=True)

from depthai_sdk import OakCamera

with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=60)
    oak.visualize([stereo.out_disparity, stereo.out_depth])
    oak.start(blocking=True)

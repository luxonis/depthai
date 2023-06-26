from depthai_sdk import OakCamera

with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=60)

    stereo.config_output(depth_score=True)

    oak.visualize(stereo.out.depth, fps=True).stereo(depth_score=True)
    oak.start(blocking=True)

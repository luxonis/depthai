from depthai_sdk import OakCamera
import depthai as dai


with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=30, encode='h264')

    # Set on-device output colorization, works only for encoded output
    stereo.set_colormap(dai.Colormap.JET)

    oak.visualize(stereo.out.encoded, fps=True)
    oak.start(blocking=True)

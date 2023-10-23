from depthai_sdk import OakCamera
import depthai as dai


with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=30)
    stereo.set_colormap(dai.Colormap.JET) # Must be set before creating the encoder
    encoder = oak.create_encoder(stereo, codec='h264')

    oak.visualize(encoder, fps=True)
    oak.start(blocking=True)

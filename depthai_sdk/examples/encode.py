from depthai_sdk import OakCamera, RecordType


with OakCamera() as oak:
    # color = oak.create_camera('color', fps=15)
    # left = oak.create_camera('left', fps=15)
    # right = oak.create_camera('right', fps=15)
    stereo = oak.create_stereo(resolution='800P')

    oak.visualize(stereo.out_depth, scale=2/3, fps=True)
    oak.record(stereo.out_depth, './', RecordType.MCAP)

    oak.build() # Required to access oak.device
    oak.device.setIrLaserDotProjectorBrightness(1000)

    # oak.show_graph()
    oak.start(blocking=True)
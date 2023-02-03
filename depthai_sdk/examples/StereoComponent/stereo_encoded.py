from depthai_sdk import OakCamera

with OakCamera() as oak:
    stereo = oak.create_stereo('400p', fps=30, encode='h264')
    # oak.visualize(stereo.out.encoded)
    oak.callback(stereo.out.encoded, callback=lambda x: print(x))
    oak.start(blocking=True)

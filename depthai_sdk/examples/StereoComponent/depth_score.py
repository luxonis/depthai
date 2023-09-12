from depthai_sdk import OakCamera

def callback(packet):
    print(packet.depth_score)

with OakCamera() as oak:
    stereo = oak.create_stereo('800p', fps=60)

    stereo.config_output(depth_score=True)
    stereo.config_output(depth_score=True)
    oak.callback(stereo.out.disparity, callback)
    oak.start(blocking=True)

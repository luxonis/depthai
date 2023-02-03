from depthai_sdk import OakCamera

with OakCamera() as oak:
    left = oak.create_camera('left', resolution='400p', fps=60, encode='h264')

    # oak.visualize([left, right], fps=True)
    oak.callback(left, callback=lambda packet: print(packet.imgFrame.getSequenceNum()))
    oak.start(blocking=True)

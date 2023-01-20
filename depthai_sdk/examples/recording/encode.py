from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=20, encode='H265')
    left = oak.create_camera('left', resolution='800p', fps=20, encode='H265')
    right = oak.create_camera('right', resolution='800p', fps=20, encode='H265')

    stereo = oak.create_stereo(left=left, right=right)
    nn = oak.create_nn('mobilenet-ssd', color, spatial=stereo)

    # Sync & save all (encoded) streams
    oak.record([color.out.encoded, left.out.encoded, right.out.encoded], './', RecordType.VIDEO)
    oak.visualize([nn])

    oak.start(blocking=True)

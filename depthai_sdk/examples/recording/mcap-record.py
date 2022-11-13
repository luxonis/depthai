from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=20, encode='MJPEG')
    color.config_color_camera(ispScale=(2,3)) # 720P
    left = oak.create_camera('left', resolution='720p', fps=20, encode='MJPEG')
    right = oak.create_camera('right', resolution='720p', fps=20, encode='MJPEG')
    stereo = oak.create_stereo(left=left, right=right)

    # Sync & save all (encoded) streams
    recorder = oak.record([color.out.encoded, left.out.encoded, right.out.encoded, stereo.out.depth], './', RecordType.MCAP)
    oak.visualize(left)
    oak.start(blocking=True)
from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=30, encode='MJPEG')
    color.config_color_camera(isp_scale=(2, 3)) # 720P
    left = oak.create_camera('left', resolution='400p', fps=30)
    right = oak.create_camera('right', resolution='400p', fps=30)
    stereo = oak.create_stereo(left=left, right=right)

    # Sync & save all streams
    recorder = oak.record([color.out.encoded, left, right, stereo.out.depth], './', RecordType.MCAP)
    # recorder.config_mcap(pointcloud=True)
    oak.visualize(left)
    oak.start(blocking=True)
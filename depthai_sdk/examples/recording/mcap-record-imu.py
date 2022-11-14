from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    left = oak.create_camera('left', resolution='400p', fps=30)
    right = oak.create_camera('right', resolution='400p', fps=30)
    stereo = oak.create_stereo(left=left, right=right)

    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    recorder = oak.record([stereo.out.depth, imu], './', RecordType.MCAP)

    oak.visualize(left)
    oak.start(blocking=True)
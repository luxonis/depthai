from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    left = oak.create_camera('left', resolution='400p', fps=30)
    right = oak.create_camera('right', resolution='400p', fps=30)
    stereo = oak.create_stereo(left=left, right=right)

    imu = oak.create_imu()
    imu.config_imu(report_rate=500, batch_report_threshold=5)

    # Note that for MCAP recording, user has to have ROS installed
    recorder = oak.record([imu, stereo.out.depth], './', RecordType.MCAP)

    oak.visualize([left, stereo])
    oak.start(blocking=True)
from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    left = oak.create_camera('left', resolution='800p', encode='jpeg', fps=30)
    right = oak.create_camera('right', resolution='800p', encode='jpeg', fps=30)
    stereo = oak.create_stereo(left=left, right=right)
    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    # DB3 / ROSBAG
    oak.record([left.out.encoded, right.out.encoded, stereo.out.depth, imu], 'record', record_type=RecordType.DB3)

    # Record left only
    oak.visualize(left.out.encoded)
    oak.start(blocking=True)

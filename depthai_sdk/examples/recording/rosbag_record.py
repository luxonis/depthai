from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    color = oak.create_camera('color', fps=30)
    left = oak.create_camera('left', resolution='800p', fps=30)
    right = oak.create_camera('right', resolution='800p', fps=30)
    color_encoder = oak.create_encoder(color, codec='mjpeg')
    left_encoder = oak.create_encoder(left, codec='mjpeg')
    right_encoder = oak.create_encoder(right, codec='mjpeg')
    stereo = oak.create_stereo(left=left, right=right)
    stereo.config_stereo(align=color)
    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    # DB3 / ROSBAG. ROSBAG doesn't require having ROS installed, while DB3 does.
    record_components = [left_encoder, color_encoder, right_encoder, stereo.out.depth, imu]
    oak.record(record_components, 'record', record_type=RecordType.ROSBAG)

    # Visualize only color stream
    oak.visualize(color_encoder)
    oak.start(blocking=True)

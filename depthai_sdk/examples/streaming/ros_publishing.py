from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', encode='jpeg', fps=30)
    color.config_color_camera(isp_scale=(2,3))
    left = oak.create_camera('left', resolution='400p', encode='jpeg',fps=30)
    right = oak.create_camera('right', resolution='400p', encode='jpeg',fps=30)
    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    oak.ros_stream([left, right, color, imu])
    # oak.visualize(left)
    oak.start(blocking=True)

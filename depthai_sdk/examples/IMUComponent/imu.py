from depthai_sdk import OakCamera

with OakCamera() as oak:
    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)
    # DepthAI viewer should open, and IMU data can be viewed on the right-side panel,
    # under "Stats" tab (right of the "Device Settings" tab).
    oak.visualize(imu.out.main)
    oak.start(blocking=True)

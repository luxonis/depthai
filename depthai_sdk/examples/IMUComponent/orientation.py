from contourpy.chunk import calc_chunk_sizes
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import IMUPacket, FramePacket
import depthai as dai
import rerun as rr
from ahrs.filters import Mahony
import subprocess
import numpy as np

mahony = Mahony(frequency=100)
mahony.Q = np.array([1, 0, 0, 0], dtype=np.float64)

"""
TODO:
- SR, LR, OAK-D,
OAK-D Lite

FFC-4P FFC-3P - can't, modular cameras
"""
imu_rotation_data = {
    'cameras': [
        {
            'devices': ['OAK-D-S2', 'OAK-D-W', 'OAK-D-PRO', 'OAK-D Pro-W'],
            'boardRev': 'R', # Any revision
            'BNO086': {'roll': 90, 'pitch': 90, 'yaw': 0},
            'BMI270': {'roll': 90, 'pitch': 90, 'yaw': 0},
        },
        {
            'devices': ['OAK-D-S2-POE', 'OAK-D POE-W', 'OAK-D-PRO-POE', 'OAK-D-PRO-W-POE'],
            'boardRev': 'R4', # Single PCB version (2022)
            # TODO test BMO, should be +180deg yaw I think
            'BMI270': {'roll': 90, 'pitch': 90, 'yaw': 90},
        },
        {
            'devices': ['OAK-D-S2-POE', 'OAK-D POE-W', 'OAK-D-PRO-POE', 'OAK-D-PRO-W-POE'],
            'boardRev': 'R5', # Multi-PCB version (2023), BNO/BMI should have same coordinate system, same as BNO on R4
            'BNO086': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'BMI270': {'roll': 0, 'pitch': 0, 'yaw': 0},
        },
        {
            'devices': ['OAK-D-LITE'],
            'boardRev': 'R', # Any revision
            'BNO086': {'roll': 90, 'pitch': 90, 'yaw': 0},
            # 'BMI270': {'roll': 90, 'pitch': 90, 'yaw': 0},
        },
        {
            'devices': ['OAK-D-POE', 'OAK-D-CM4-POE'],
            'boardRev': 'R', # Any revision
            'BNO086': {'roll': 90, 'pitch': 90, 'yaw': 0},
            # BMI??
        },
        # OAK-D should be at the bottom, as we are checking if device name starts with it
    ]
}

def imu_rotation(device: dai.Device):
    imu_name = device.getConnectedIMU()
    device_name: str = device.getDeviceName()
    eeprom = device.readCalibration().getEepromData()
    rev = eeprom.boardRev
    print(device_name, imu_name, rev)

    def calculate_rotation(rotation):
        def rotation_matrix(axis, angle_degree):
            angle = np.deg2rad(angle_degree)
            if axis == 'roll': # x
                return np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]
                ])
            elif axis == 'pitch': # y
                return np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
            elif axis == 'yaw': # z
                return np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])

        return np.dot(
            rotation_matrix('roll', rotation['roll']),
            rotation_matrix('pitch', rotation['pitch']),
            rotation_matrix('yaw', rotation['yaw'])
        )

    for camera in imu_rotation_data['cameras']:
        for name in camera['devices']:
            if device_name.startswith(name) and rev.startswith(camera['boardRev']):
                if imu_name == 'BNO086':
                    return calculate_rotation(camera['BNO086'])
                elif imu_name == 'BMI270':
                    return calculate_rotation(camera['BMI270'])
    raise Exception(f"IMU rotation not found! Please report this issue to Luxonis forum. Device name '{device_name}', IMU name '{imu_name}', board revision '{rev}'")

with OakCamera() as oak:
    subprocess.Popen(["rerun","--memory-limit", "20MB"])
    rr.init("Plot test")
    rr.connect()
    imu = oak.create_imu()
    # imu.node.enableFirmwareUpdate(True)

    rotation = imu_rotation(oak.device)
    exit()

    imu.config_imu(report_rate=400, batch_report_threshold=1)
    color = oak.create_camera("color", dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    intrinsics = oak.device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(1920, 1080))

    def imu_callback(packet: IMUPacket):
        for d in packet.data:
            gyro: dai.IMUReportGyroscope = d.gyroscope
            accel: dai.IMUReportAccelerometer = d.acceleroMeter
            gryo_rotated = np.dot(rotation, np.array([gyro.x, gyro.y, gyro.z]))
            accel_rotated = np.dot(rotation, np.array([accel.x, accel.y, accel.z]))
            mahony.Q = mahony.updateIMU(mahony.Q, gryo_rotated, accel_rotated)

    def color_callback(packet: FramePacket):
        rr.log_pinhole("world/camera/image", child_from_parent=intrinsics, width=1920, height=1080)
        rr.log_image("world/camera/image/rgb", packet.frame[:, :, ::-1]) # BGR to RGB
        rr.log_rigid3("world/camera", child_from_parent=([0,0,0], mahony.Q))

    oak.callback(imu.out.main, callback=imu_callback)
    oak.callback(color.out.camera, callback=color_callback)
    oak.start(blocking=True)
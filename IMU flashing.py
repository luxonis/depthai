import depthai as dai
import cv2
import numpy as np
import open3d as o3d
import json
from pathlib import Path
from depthai_sdk import OakCamera
from depthai_sdk.components.stereo_component import WLSLevel
from depthai_sdk.visualize.configs import StereoColor
from depthai_sdk.components.parser import parse_resolution, parse_encode, parse_camera_socket
from scipy.spatial.transform import Rotation as R
import numpy as np

def str_to_cam(cam: str) -> dai.CameraBoardSocket:
	return dai.CameraBoardSocket.__members__[cam]

with OakCamera() as oak:
    oak.calibration = oak._oak.device.readCalibration()
    eeprom = oak.calibration.getEepromData()
    detection = eeprom.boardName.split()
    eeprom.productName = eeprom.productName.replace(" ", "-")
    eeprom.boardName = eeprom.boardName.replace(" ", "-")
    print(f"Product name: {eeprom.productName}, board name {eeprom.boardName}")
    if eeprom.productName.split("-")[0] == "OAK":
        detection = eeprom.productName.split("-")
    elif eeprom.boardName.split("-")[0] == "OAK":
        detection = eeprom.boardName.split("-")
    if "AF" in detection:
        detection.remove("AF")
    if "FF" in detection:
        detection.remove("FF")
    if "9782" in detection:
        detection.remove("9782")
    oak.board_name = '-'.join(detection).upper()
    board_path = Path(oak.board_name)
    if not board_path.exists():
        board_path = (Path(__file__).parent / 'resources/depthai-boards/boards' / oak.board_name.upper()).with_suffix('.json').resolve()
    if not board_path.exists():
        raise ValueError(
            'Board config not found: {}'.format(board_path))
    with open(board_path) as fp:
        oak.board_config = json.load(fp)
        oak.board_config = oak.board_config['board_config']
        oak.board_config_backup = oak.board_config
    board_config_dict = oak.board_config
    if "imuExtrinsics" in board_config_dict.keys():
        string= "boardConf"
        print(f"Detecting the IMU sensor from boardConfig {oak.calibration.eepromToJson()[string]}")
        if oak.calibration.eepromToJson()["boardConf"] !="":
            if oak.calibration.eepromToJson()["boardConf"][-2:] == "00":
                board_config_dict = board_config_dict["imuExtrinsics"]["sensors"]["BNO"]
                print("Detected BNO085.")
            if oak.calibration.eepromToJson()["boardConf"][-2:]== "01":
                board_config_dict = board_config_dict["imuExtrinsics"]["sensors"]["BMI"]
                print("Detected BMI270.")
        else:
            if board_config_dict["imuExtrinsics"]["sensors"]["BMI"]:
                board_config_dict = board_config_dict["imuExtrinsics"]["sensors"]["BMI"]
            else:
                board_config_dict = board_config_dict["imuExtrinsics"]["sensors"]["BNO"]
        IMU_spec_translation = [
        board_config_dict["extrinsics"]["specTranslation"]['x'], 
            board_config_dict["extrinsics"]["specTranslation"]['y'], 
            board_config_dict["extrinsics"]["specTranslation"]['z']]
        rot = board_config_dict["extrinsics"]['rotation']
        rotation = R.from_euler('zyx', [rot['y'], rot['p'], rot['r']], degrees=True).as_matrix().astype(np.float32)
        oak.calibration.setImuExtrinsics(str_to_cam(board_config_dict["extrinsics"]["to_cam"]), rotation, IMU_spec_translation, IMU_spec_translation)
    eeepromData = oak.calibration.getEepromData()
    updatedCalib = dai.CalibrationHandler(eeepromData)
    oak._oak.device.flashCalibration2(oak.calibration)
    savefile = str(oak.board_name) + ".json"
    print(f"Calibration flashed in {Path(__file__).parent  / savefile}")
    updatedCalib.eepromToJsonFile(Path(__file__).parent  / savefile)
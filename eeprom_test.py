import depthai as dai
import json, time

pipeline = dai.Pipeline()

device = dai.Device()
calib_path = '/home/sachin/Desktop/luxonis/depthai/batch/Leonardo/eeprom_oak_d_pro_w_poe.json'
with open(calib_path) as jfile:
    eepromDataJson = json.load(jfile)
    calib_data = dai.CalibrationHandler.fromJson(eepromDataJson)

    device.flashFactoryCalibration(calib_data)
    device.flashCalibration2(calib_data)

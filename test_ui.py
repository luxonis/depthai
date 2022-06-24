import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QRunnable, pyqtSlot, pyqtSignal, QThreadPool

import sys, traceback
from datetime import datetime
# from PyQt5.QtWidgets import QMessageBox
# from PyQt5.QtWidgets import QMessageBox
# import numpy as np
import depthai as dai
import argparse
import os
# import blobconverter
import signal
import json, time
from pathlib import Path
import cv2

# Try setting native cv2 image format, otherwise RGB888
colorMode = QtGui.QImage.Format_RGB888
try:
    colorMode = QtGui.QImage.Format_BGR888
except:
    colorMode = QtGui.QImage.Format_RGB888

FPS = 10

BATCH_DIR = Path(__file__).resolve().parent / 'batch'

test_result = {
    'usb3_res': '',
    'eeprom_res': '',
    'rgb_cam_res': '',
    'jpeg_enc_res': '',
    'prew_out_rgb_res': '',
    'left_cam_res': '',
    'right_cam_res': '',
    'left_strm_res': '',
    'right_strm_res': '',
    'eeprom_data': '',
    'nor_flash_res': '',
}
OAK_KEYS = {
    'OAK-1': ['usb3_res', 'rgb_cam_res', 'jpeg_enc_res', 'prew_out_rgb_res'],
    'OAK-D': ['usb3_res', 'rgb_cam_res', 'jpeg_enc_res', 'prew_out_rgb_res', 'left_cam_res', 'right_cam_res','left_strm_res', 'right_strm_res'],
    'OAK-D-PRO-POE': ['usb3_res', 'rgb_cam_res', 'jpeg_enc_res', 'prew_out_rgb_res', 'left_cam_res', 'right_cam_res','left_strm_res', 'right_strm_res', 'eeprom_data', 'nor_flash_res'],
}

operator_tests = {
    'jpeg_enc': '',
    'prew_out_rgb': '',
    'left_strm': '',
    'right_strm': '',
    'ir_light': ''
}
OP_OAK_KEYS = {
    'OAK-1': ['jpeg_enc', 'prew_out_rgb'],
    'OAK-D': ['jpeg_enc', 'prew_out_rgb', 'left_strm', 'right_strm'],
    'OAK-D-PRO': ['jpeg_enc', 'prew_out_rgb', 'left_strm', 'right_strm', 'ir_light'],
    'OAK-D-PRO-POE': ['jpeg_enc', 'prew_out_rgb', 'left_strm', 'right_strm', 'ir_light'],
}


OAK_D_LABELS = '<html><head/><body><p align=\"right\"><span style=\" font-size:14pt;\"> \
        USB3 <br style="font-size:18pt"> \
        EEPROM write test <br style="font-size:22pt"> \
        RGB Camera connected  <br style="font-size:21pt"> \
        JPEG Encoding Stream <br style="font-size:21pt"> \
        preview-out-rgb Stream <br style="font-size:21pt"> \
        left camera connected <br style="font-size:23pt"> \
        right camera connected<br style="font-size:21pt"> \
        left Stream <br style="font-size:22pt"> \
        right Stream <br style="font-size:21pt"> </span></p></body></html>'


OAK_ONE_LABELS = '<html><head/><body><p align=\"right\"><span style=\" font-size:14pt;\"> \
        USB3 <br style="font-size:18pt"> \
        EEPROM write test <br style="font-size:22pt"> \
        RGB Camera connected  <br style="font-size:21pt"> \
        JPEG Encoding Stream <br style="font-size:21pt"> \
        preview-out-rgb Stream <br style="font-size:21pt"></span></p></body></html>'

CSV_HEADER = {
    'OAK-1': '"Device ID","Device Type","Timestamp","USB3","RGB camera connect","JPEG Encoding","RGB Stream","JPEG Encoding Operator","RGB Encoding Operator"',
    'OAK-D': '"Device ID","Device Type","Timestamp","USB3","RGB camera connect","JPEG Encoding","RGB Stream","Left camera connect","Right camera connect","Left Stream","Right Stream","RGB Stream Operator","JPEG Encoding Operator","Left Stream Operator","Right Stream Operator"',
    'OAK-D-PRO': '"Device ID","Device Type","Timestamp","USB3","RGB camera connect","JPEG Encoding","RGB Stream","Left camera connect","Right camera connect","Left Stream","Right Stream","RGB Stream Operator","JPEG Encoding Operator","Left Stream Operator","Right Stream Operator","IR Light"',
    'OAK-D-PRO-POE': '"Device ID","Device Type","Timestamp","USB3","RGB camera connect","JPEG Encoding","RGB Stream","Left camera connect","Right camera connect","Left Stream","Right Stream","EEPROM DATA","NOR FLASH","RGB Stream Operator","JPEG Encoding Operator","Left Stream Operator","Right Stream Operator","IR Light"',
}


def set_operator_test(test):
    global operator_tests
    if test.isChecked():
        operator_tests[test.name] = test.value
        # print(test.name + ' ' + test.value)


update_res = False
prew_width = 0
prew_height = 0


def clear_test_results():
    global update_res
    for key in test_result:
        test_result[key] = ''
    for key in operator_tests:
        operator_tests[key] = ''
    update_res = True

imu_upgrade = True
class DepthAICamera():
    def __init__(self):
        global update_res
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRgb.setStreamName("rgb")
        self.camRgb.setPreviewSize(300, 300)
        self.camRgb.setPreviewKeepAspectRatio(True)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        self.camRgb.preview.link(self.xoutRgb.input)
        self.camRgb.setFps(FPS)

        # # TMP TMP TMP - IMU update simulation
        # script = self.pipeline.create(dai.node.Script)
        # script.setScript("""
        #     import time

        #     status = False

        #     if status:
        #         progress = 3
        #         while progress <= 100:
        #             node.warn(f'IMU firmware update status: {progress}%')
        #             progress += 1.5
        #             time.sleep(0.06)
        #         time.sleep(1)
        #         node.info("IMU firmware update succesful!");
        #     else:
        #         time.sleep(2)
        #         node.error("IMU firmware update failed. Your board likely doesn't have IMU!")
        #         time.sleep(5)
        # """)

        # Add IMU to force FW update
        if imu_upgrade:
            self.imu = self.pipeline.create(dai.node.IMU)
            self.xoutIMU = self.pipeline.create(dai.node.XLinkOut)
            self.xoutIMU.setStreamName("IMU")
            if 'lite' in test_type:
                self.imu.enableFirmwareUpdate(True)
            self.imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
            self.imu.setBatchReportThreshold(1)
            self.imu.setMaxBatchReports(10)
            self.imu.out.link(self.xoutIMU.input)



        self.videoEnc = self.pipeline.create(dai.node.VideoEncoder)
        self.camRgb.video.link(self.videoEnc.input)
        self.xoutJpeg = self.pipeline.create(dai.node.XLinkOut)
        self.videoEnc.bitstream.link(self.xoutJpeg.input)
        self.videoEnc.setDefaultProfilePreset(self.camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
        self.xoutJpeg.setStreamName("jpeg")

        if test_type != 'OAK-1':
            self.camLeft = self.pipeline.create(dai.node.MonoCamera)
            self.xoutLeft = self.pipeline.create(dai.node.XLinkOut)
            self.xoutLeft.setStreamName("left")
            self.camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
            self.camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            self.camLeft.out.link(self.xoutLeft.input)
            self.camLeft.setFps(FPS)

            self.camRight = self.pipeline.create(dai.node.MonoCamera)
            self.xoutRight = self.pipeline.create(dai.node.XLinkOut)
            self.xoutRight.setStreamName("right")
            self.camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            self.camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            self.camRight.out.link(self.xoutRight.input)
            self.camRight.setFps(10)

        usb_speed = dai.UsbSpeed.SUPER
        # if test_type == 'OAK-D-PRO-POE':
        if 'poe' in eepromDataJson['productName'].lower():
            usb_speed = dai.UsbSpeed.HIGH

        self.device = dai.Device(dai.OpenVINO.VERSION_2021_4, usb_speed)

        # Check cameras, if center is smaller, modify all to be same (all cams OV case)
        # cams = self.device.getConnectedCameraProperties()
        # for cam in cams:
        #     if cam.socket == dai.CameraBoardSocket.CENTER:
        #         print(f'Center camera w/h: ({cam.width}, {cam.height})')
        #         if cam.height == 800:
        #             self.camRgb.setPreviewSize(cam.width, cam.height)
        #             self.camRgb.setVideoSize(cam.width, cam.height)
        #             self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        #             if test_type != 'OAK-1':
        #                 self.camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        #                 self.camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        #         elif cam.height == 720:
        #             self.camRgb.setPreviewSize(cam.width, cam.height)
        #             self.camRgb.setVideoSize(cam.width, cam.height)
        #             self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        #             if test_type != 'OAK-1':
        #                 self.camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #                 self.camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        self.device.startPipeline(self.pipeline)

        # if test_type == 'OAK-D-PRO' or test_type == 'OAK-D-PRO-POE':
        if 'nir' not in eepromDataJson['boardConf'] and 'ir' in eepromDataJson['boardConf']:
            try:
                success = True
                success = success and self.device.setIrLaserDotProjectorBrightness(100)
                print(f'set laser dot, result {success}')
                success = success and self.device.setIrFloodLightBrightness(250)
                print(f'set flood light, result {success}')
                if not success:
                    raise Exception
            except:
                print('IR sensor not working!')

        cameras = self.device.getConnectedCameras()
        if dai.CameraBoardSocket.RGB not in cameras:
            test_result['rgb_cam_res'] = 'FAIL'
        else:
            test_result['rgb_cam_res'] = 'PASS'
        if dai.CameraBoardSocket.LEFT not in cameras:
            test_result['left_cam_res'] = 'FAIL'
        else:
            test_result['left_cam_res'] = 'PASS'
        if dai.CameraBoardSocket.RIGHT not in cameras:
            test_result['right_cam_res'] = 'FAIL'
        else:
            test_result['right_cam_res'] = 'PASS'

        update_res = True

        speed = self.device.getUsbSpeed().name
        print('Usb speed: ', speed)
        try:
            if speed == 'SUPER' or speed == 'SUPER_PLUS':
                test_result['usb3_res'] = 'PASS'
            else:
                test_result['usb3_res'] = 'FAIL'
        except RuntimeError:
            test_result['usb3_res'] = 'FAIL'

        # if test_type == 'OAK-D-PRO-POE':
        if 'poe' in eepromDataJson['productName'].lower():
            test_result['usb3_res'] = 'SKIP'

        self.start_queue()
        self._rgb_pass = 0
        self._left_pass = 0
        self._right_pass = 0
        self._NR_TEST_FRAMES = 40
        self._FRAME_WAIT = FPS*8
        self._FRAMES_WAIT = FPS*3
        self._rgb_timer = 0
        self._left_timer = 0
        self._right_timer = 0
        self._FRAME_JPEG = 10
        self.current_jpeg = 0

        self.id = self.device.getDeviceInfo().getMxId()

    def __del__(self):
        self.device.close()

    def start_queue(self):
        global update_res
        try:
            self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        except RuntimeError:
            test_result['prew_out_rgb_res'] = 'FAIL'
        try:
            self.qLeft = self.device.getOutputQueue(name="left", maxSize=4, blocking=False)
        except RuntimeError:
            test_result['left_strm_res'] = 'FAIL'
        try:
            self.qRight = self.device.getOutputQueue(name='right', maxSize=4, blocking=False)
        except RuntimeError:
            test_result['right_strm_res'] = 'FAIL'
        try:
            self.qJpeg = self.device.getOutputQueue(name="jpeg", maxSize=1, blocking=False)
        except RuntimeError:
            test_result['jpeg_enc_res'] = 'FAIL'
        update_res = True

    def get_image(self, cam_type):
        global update_res
        image = None
        try:
            if cam_type == 'RGB':
                if test_result['rgb_cam_res'] == 'PASS':
                    in_rgb = self.qRgb.tryGet()
                    if in_rgb is not None:
                        image = in_rgb.getCvFrame()
                        if colorMode == QtGui.QImage.Format_RGB888:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if test_result['prew_out_rgb_res'] == '':
                        if (self._rgb_timer > self._FRAME_WAIT) or (self._rgb_timer > self._FRAME_WAIT and self._rgb_pass == 0):
                            test_result['prew_out_rgb_res'] = 'FAIL'
                            update_res = True
                        elif self._rgb_pass == self._NR_TEST_FRAMES:
                            test_result['prew_out_rgb_res'] = 'PASS'
                            update_res = True
                        elif image is not None:
                            self._rgb_pass += 1
                        self._rgb_timer += 1
            if cam_type == 'LEFT':
                if test_result['left_cam_res'] == 'PASS':
                    in_left = self.qLeft.tryGet()
                    if in_left is not None:
                        image = in_left.getCvFrame()
                    if test_result['left_strm_res'] == '':
                        if (self._left_timer > self._FRAME_WAIT) or (self._left_timer > self._FRAME_WAIT and self._left_pass == 0):
                            test_result['left_strm_res'] = 'FAIL'
                            update_res = True
                        elif self._left_pass == self._NR_TEST_FRAMES:
                            test_result['left_strm_res'] = 'PASS'
                            update_res = True
                        elif image is not None:
                            self._left_pass += 1
            if cam_type == 'RIGHT':
                if test_result['right_cam_res'] == 'PASS':
                    in_right = self.qRight.tryGet()
                    if in_right is not None:
                        image = in_right.getCvFrame()
                    if test_result['right_strm_res'] == '':
                        if (self._right_timer > self._FRAME_WAIT) or (self._right_timer > self._FRAME_WAIT and self._right_pass == 0):
                            test_result['right_strm_res'] = 'FAIL'
                            update_res = True
                        elif self._right_pass == self._NR_TEST_FRAMES:
                            test_result['right_strm_res'] = 'PASS'
                            update_res = True
                        elif image is not None:
                            self._right_pass += 1
            if cam_type == 'JPEG':
                if test_result['jpeg_enc_res'] == '' and test_result['rgb_cam_res'] != 'FAIL':
                    in_jpeg = self.qJpeg.tryGet()
                    # print(in_jpeg)
                    if in_jpeg is not None:
                        image = in_jpeg.getData()
                        self.current_jpeg += 1
                        if self.current_jpeg > 10:
                            test_result['jpeg_enc_res'] = 'PASS'
                            update_res = True
                            print(image)
                            # for encFrame in qJpeg.tryGetAll():
                            #     with open(f"{dirName}/{int(time.time() * 1000)}.jpeg", "wb") as f:
                            #         f.write(bytearray(encFrame.getData()))
                            return True, image
                    return False, image
                pass
        except RuntimeError:
            if cam_type == 'RGB' and self._rgb_pass < self._NR_TEST_FRAMES:
                test_result['prew_out_rgb_res'] = 'FAIL'
            if cam_type == 'LEFT' and self._left_pass < self._NR_TEST_FRAMES:
                test_result['left_strm_res'] = 'FAIL'
            if cam_type == 'RIGHT' and self._right_pass < self._NR_TEST_FRAMES:
                test_result['right_cam_res'] = 'FAIL'
            if cam_type == 'JPEG' and self.current_jpeg > self._FRAME_JPEG:
                test_result['jpeg_enc_res'] = 'FAIL'
            update_res = True
            return False, None
        return True, image

    def flash_eeprom(self):
        global eepromDataJson
        try:
            device_calib = self.device.readCalibration()
            device_calib.eepromToJsonFile(CALIB_BACKUP_FILE)
            print('Calibraton Data on the device is backed up at: ', CALIB_BACKUP_FILE, sep='\n')

            # Opening JSON file
            if eepromDataJson is None:
                return False

            eepromDataJson['batchTime'] = int(time.time())

            calib_data = dai.CalibrationHandler.fromJson(eepromDataJson)

            # Flash both factory & user areas
            self.device.flashFactoryCalibration(calib_data)
            self.device.flashCalibration2(calib_data)
        except Exception as ex:
            errorMsg = f'Calibration Flash Failed: {ex}'
            print(errorMsg)
            return (False, errorMsg, eepromDataJson)

        print('Calibration Flash Successful')
        return (True, '', eepromDataJson)


eepromDataJson = None
calib_path = None
class Ui_CalibrateSelect(QtWidgets.QDialog):
    def __init__(self):
        global eepromDataJson
        global calib_path
        super().__init__()
        x = os.walk(BATCH_DIR)
        print(x)
        self.batches = []
        self.jsons = {}
        for x in os.walk(BATCH_DIR):
            self.batches = sorted(x[1])
            break
        for batch in self.batches:
            for js in os.walk(BATCH_DIR/batch):
                self.jsons[batch] = sorted(js[2])
                print(js)

        self.batch_combo = QtWidgets.QComboBox(self)
        self.batch_combo.setGeometry(QtCore.QRect(100, 20, 141, 32))
        self.batch_combo.addItems(self.batches)
        self.batch_combo.currentTextChanged.connect(self.batches_changed)
        self.json_combo = QtWidgets.QComboBox(self)
        self.json_combo.setGeometry(QtCore.QRect(100, 70, 261, 32))
        self.json_combo.addItems(self.jsons[self.batches[0]])
        self.json_combo.currentTextChanged.connect(self.json_changed)

        self.setObjectName("CalibrateSelect")
        self.resize(386, 192)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(70, 130, 191, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.batch_label = QtWidgets.QLabel(self)
        self.batch_label.setGeometry(QtCore.QRect(10, 20, 51, 27))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.batch_label.setFont(font)
        self.batch_label.setObjectName("batch_label")
        self.json_label = QtWidgets.QLabel(self)
        self.json_label.setGeometry(QtCore.QRect(10, 70, 77, 27))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.json_label.setFont(font)
        self.json_label.setObjectName("json_label")
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)

        _translate = QtCore.QCoreApplication.translate
        self.batch_label.setText(_translate("CalibrateSelect", "Batch"))
        self.json_label.setText(_translate("CalibrateSelect", "JSON file"))
        self.setWindowTitle(_translate("CalibrateSelect", "Dialog"))
        calib_path = BATCH_DIR / self.batch_combo.currentText() / self.json_combo.currentText()
        with open(calib_path) as jfile:
            eepromDataJson = json.load(jfile)

    def batches_changed(self):
        global calib_path
        global eepromDataJson
        print("Batches changed!")
        self.json_combo.clear()
        self.json_combo.addItems(self.jsons[self.batch_combo.currentText()])
        calib_path = BATCH_DIR / self.batch_combo.currentText() / self.json_combo.currentText()
        with open(calib_path) as jfile:
            eepromDataJson = json.load(jfile)

    def json_changed(self):
        global eepromDataJson
        global calib_path
        calib_path = BATCH_DIR / self.batch_combo.currentText() / self.json_combo.currentText()
        with open(calib_path) as jfile:
            eepromDataJson = json.load(jfile)


class Camera(QtWidgets.QWidget):
    def __init__(self, get_image, camera_format, title='Camera', location=(0, 0)):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon('Assets/logo.png'))
        width, height = location
        self.move(width, height)
        self.camera = QtWidgets.QLabel('Camera')
        self.camera.setFixedSize(prew_width, prew_height)
        self.camera.resize(prew_width, prew_height)
        layout.addWidget(self.camera)
        self.setLayout(layout)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(1000//FPS)
        self.get_image = get_image
        self.camera_format = camera_format

    def update_image(self):
        status, image = self.get_image()
        if status and image is not None:
            if len(image.shape) > 1:
                q_image = QtGui.QImage(image.data, image.shape[1], image.shape[0], self.camera_format)
                pixmap = QtGui.QPixmap.fromImage(q_image)
            else:
                pixmap = QtGui.QPixmap()
                pixmap.loadFromData(image)
            pixmap = pixmap.scaled(prew_width, prew_height, QtCore.Qt.KeepAspectRatio)
            self.camera.setPixmap(pixmap)
        # else:
        #     # print('im hiding')
        #     self.hide()


WIDTH = 766
HEIGHT = 717


def test_connexion():
    result = False
    try:
        # while not result:
        (result, info) = dai.DeviceBootloader.getFirstAvailableDevice()
    finally:
        return result

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error:`tuple` (exctype, value, traceback.format_exc() )
    - result: `object` data returned from processing, anything
    - progress: `tuple` indicating progress metadata
    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(tuple)


class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    '''
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._run = True

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = False
            while self._run and not result:
                result = self.fn(*self.args, **self.kwargs)
                time.sleep(0.1)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.result.emit(result)  # Return the result of the processing
            self.signals.finished.emit()  # Done

    def stop(self):
        self._run = False


# BW compat...
UI_tests = None
class UiTests(QtWidgets.QMainWindow):
    print_logs_trigger = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.MB_INIT = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "</style></head><body style=\" font-family:\'Sans Serif\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
        self.MB_END = "</p></body></html>"
        self.all_logs = ""

        x = os.walk(BATCH_DIR)
        print(x)
        self.batches = []
        self.jsons = {}
        i = 0
        for x in os.walk(BATCH_DIR):
            if i == 0:
                self.batches = x[1]
            else:
                self.jsons[self.batches[i - 1]] = x[2]
            i = i + 1
        dialog = Ui_CalibrateSelect()
        if not dialog.exec_():
            sys.exit()

        self.print_logs_trigger.connect(self.print_logs)

    def setupUi(self):
        global UI_tests
        UI_tests = self

        UI_tests.closeEvent = self.close_event
        UI_tests.setObjectName("UI_tests")
        UI_tests.resize(WIDTH, HEIGHT)
        UI_tests.move(0, 0)
        UI_tests.setWindowTitle("DepthAI UI Tests")
        UI_tests.setWindowIcon(QtGui.QIcon('Assets/logo.png'))
        font = QtGui.QFont()
        font.setPointSize(13)
        UI_tests.setFont(font)
        self.centralwidget = QtWidgets.QWidget(UI_tests)
        self.centralwidget.setObjectName("centralwidget")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(10, 10, 751, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.connect_but = QtWidgets.QPushButton(self.centralwidget)
        self.connect_but.setGeometry(QtCore.QRect(460, 390, 86, 25))
        self.connect_but.setObjectName("connect_but")
        self.connect_but.clicked.connect(self.show_cameras)
        # self.save_but = QtWidgets.QPushButton(self.centralwidget)
        # self.save_but.setGeometry(QtCore.QRect(550, 390, 86, 25))
        # self.save_but.setObjectName("connect_but")
        # self.save_but.clicked.connect(save_csv)
        self.automated_tests = QtWidgets.QGroupBox(self.centralwidget)
        if test_type == 'OAK-1':
            self.automated_tests.setGeometry(QtCore.QRect(20, 70, 311, 241))
        else:
            self.automated_tests.setGeometry(QtCore.QRect(20, 70, 311, 395))
        self.automated_tests.setObjectName("automated_tests")
        self.automated_tests_labels = QtWidgets.QLabel(self.automated_tests)
        self.automated_tests_labels.setGeometry(QtCore.QRect(10, 20, 221, 351))
        self.automated_tests_labels.setObjectName("automated_tests_labels")
        self.automated_tests_labels.setContentsMargins(0,9,9,5)
        self.automated_tests_labels.setAlignment(QtCore.Qt.AlignRight)
        # self.automated_tests_labels.setGeometry(QtCore.QRect(10, 30, 221, 150))

        px, py, x, y = 240, 37, 51, 21
        dy = 39
        self.usb3_res = QtWidgets.QLabel(self.automated_tests)
        self.usb3_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.usb3_res.setObjectName("usb3_res")

        py += dy
        self.eeprom_res = QtWidgets.QLabel(self.automated_tests)
        self.eeprom_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.eeprom_res.setObjectName("eeprom_res")

        py += dy
        self.rgb_cam_res = QtWidgets.QLabel(self.automated_tests)
        self.rgb_cam_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.rgb_cam_res.setObjectName("rgb_cam_res")

        py += dy
        self.jpeg_enc_res = QtWidgets.QLabel(self.automated_tests)
        self.jpeg_enc_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.jpeg_enc_res.setObjectName("jpeg_enc_res")

        py += dy
        self.prew_out_rgb_res = QtWidgets.QLabel(self.automated_tests)
        self.prew_out_rgb_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.prew_out_rgb_res.setObjectName("prew_out_rgb_res")

        py += dy
        self.left_cam_res = QtWidgets.QLabel(self.automated_tests)
        self.left_cam_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.left_cam_res.setObjectName("left_cam_res")

        py += dy
        self.right_cam_res = QtWidgets.QLabel(self.automated_tests)
        self.right_cam_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.right_cam_res.setObjectName("right_cam_res")

        py += dy
        self.left_strm_res = QtWidgets.QLabel(self.automated_tests)
        self.left_strm_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.left_strm_res.setObjectName("left_strm_res")

        py += dy
        self.right_strm_res = QtWidgets.QLabel(self.automated_tests)
        self.right_strm_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.right_strm_res.setObjectName("right_strm_res")

        py += dy
        self.ir_project_res = QtWidgets.QLabel(self.automated_tests)
        self.ir_project_res.setGeometry(QtCore.QRect(px, py, x, y))
        self.ir_project_res.setObjectName("ir_project_res")

        self.operator_tests = QtWidgets.QGroupBox(self.centralwidget)
        # self.operator_tests.setGeometry(QtCore.QRect(360, 70, 321, 321))
        # if test_type == 'OAK-D-PRO' or test_type == 'OAK-D-PRO-POE':
        if 'oak-d pro' in eepromDataJson['productName'].lower():
            self.operator_tests.setGeometry(QtCore.QRect(360, 70, 321, 311))
        elif test_type == 'OAK-1':
            self.operator_tests.setGeometry(QtCore.QRect(360, 70, 321, 190))
        else:
            self.operator_tests.setGeometry(QtCore.QRect(360, 70, 321, 281))
        self.operator_tests.setObjectName("operator_tests")
        self.operator_tests_label = QtWidgets.QLabel(self.operator_tests)
        self.operator_tests_label.setGeometry(QtCore.QRect(10, 100, 131, 201))
        self.operator_tests_label.setObjectName("operator_tests_label")
        self.NOT_TESTED_LABEL = QtWidgets.QLabel(self.operator_tests)
        self.NOT_TESTED_LABEL.setGeometry(QtCore.QRect(200, 30, 61, 61))
        self.NOT_TESTED_LABEL.setObjectName("NOT_TESTED_LABEL")
        self.FAIL_LABEL = QtWidgets.QLabel(self.operator_tests)
        self.FAIL_LABEL.setGeometry(QtCore.QRect(270, 50, 41, 21))
        self.FAIL_LABEL.setObjectName("FAIL_LABEL")
        self.PASS_LABEL = QtWidgets.QLabel(self.operator_tests)
        self.PASS_LABEL.setGeometry(QtCore.QRect(150, 50, 41, 17))
        self.PASS_LABEL.setObjectName("PASS_LABEL")
        font = QtGui.QFont()
        font.setPointSize(13)

        self.op_jpeg_frame = QtWidgets.QFrame(self.operator_tests)
        self.op_jpeg_frame.setGeometry(QtCore.QRect(160, 90, 131, 41))
        self.op_jpeg_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.op_jpeg_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.op_jpeg_frame.setLineWidth(0)
        self.op_jpeg_frame.setObjectName("op_jpeg_frame")
        self.jpeg_pass_but = QtWidgets.QRadioButton(self.op_jpeg_frame)
        self.jpeg_pass_but.setEnabled(True)
        self.jpeg_pass_but.setGeometry(QtCore.QRect(10, 10, 14, 15))
        self.jpeg_pass_but.setFont(font)
        self.jpeg_pass_but.setText("")
        self.jpeg_pass_but.setObjectName("jpeg_pass_but")
        self.jpeg_pass_but.value = 'PASS'
        self.jpeg_pass_but.name = 'jpeg_enc'
        self.jpeg_pass_but.toggled.connect(lambda: set_operator_test(self.jpeg_pass_but))
        self.jpeg_ntes_but = QtWidgets.QRadioButton(self.op_jpeg_frame)
        self.jpeg_ntes_but.setEnabled(True)
        self.jpeg_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
        self.jpeg_ntes_but.setFont(font)
        self.jpeg_ntes_but.setText("")
        self.jpeg_ntes_but.setChecked(True)
        self.jpeg_ntes_but.setObjectName("jpeg_ntes_but")
        self.jpeg_ntes_but.value = ''
        self.jpeg_ntes_but.name = 'jpeg_enc'
        self.jpeg_ntes_but.toggled.connect(lambda: set_operator_test(self.jpeg_ntes_but))
        self.jpeg_fail_but = QtWidgets.QRadioButton(self.op_jpeg_frame)
        self.jpeg_fail_but.setEnabled(True)
        self.jpeg_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
        self.jpeg_fail_but.setFont(font)
        self.jpeg_fail_but.setText("")
        self.jpeg_fail_but.setObjectName("jpeg_fail_but")
        self.jpeg_fail_but.value = 'FAIL'
        self.jpeg_fail_but.name = 'jpeg_enc'
        self.jpeg_fail_but.toggled.connect(lambda: set_operator_test(self.jpeg_fail_but))

        self.op_rgb_frame = QtWidgets.QFrame(self.operator_tests)
        self.op_rgb_frame.setGeometry(QtCore.QRect(160, 140, 131, 41))
        self.op_rgb_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.op_rgb_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.op_rgb_frame.setLineWidth(0)
        self.op_rgb_frame.setObjectName("op_rgb_frame")
        self.rgb_pass_but = QtWidgets.QRadioButton(self.op_rgb_frame)
        self.rgb_pass_but.setEnabled(True)
        self.rgb_pass_but.setGeometry(QtCore.QRect(10, 10, 16, 16))
        self.rgb_pass_but.setFont(font)
        self.rgb_pass_but.setText("")
        self.rgb_pass_but.setObjectName("rgb_pass_but")
        self.rgb_pass_but.value = 'PASS'
        self.rgb_pass_but.name = 'prew_out_rgb'
        self.rgb_pass_but.toggled.connect(lambda: set_operator_test(self.rgb_pass_but))
        self.rgb_ntes_but = QtWidgets.QRadioButton(self.op_rgb_frame)
        self.rgb_ntes_but.setEnabled(True)
        self.rgb_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
        self.rgb_ntes_but.setFont(font)
        self.rgb_ntes_but.setText("")
        self.rgb_ntes_but.setChecked(True)
        self.rgb_ntes_but.setObjectName("rgb_ntes_but")
        self.rgb_ntes_but.value = ''
        self.rgb_ntes_but.name = 'prew_out_rgb'
        self.rgb_ntes_but.toggled.connect(lambda: set_operator_test(self.rgb_ntes_but))
        self.rgb_fail_but = QtWidgets.QRadioButton(self.op_rgb_frame)
        self.rgb_fail_but.setEnabled(True)
        self.rgb_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
        self.rgb_fail_but.setFont(font)
        self.rgb_fail_but.setText("")
        self.rgb_fail_but.setObjectName("rgb_fail_but")
        self.rgb_fail_but.value = 'FAIL'
        self.rgb_fail_but.name = 'prew_out_rgb'
        self.rgb_fail_but.toggled.connect(lambda: set_operator_test(self.rgb_fail_but))

        if test_type != 'OAK-1':
            self.op_left_frame = QtWidgets.QFrame(self.operator_tests)
            self.op_left_frame.setGeometry(QtCore.QRect(160, 180, 131, 41))
            self.op_left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.op_left_frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.op_left_frame.setLineWidth(0)
            self.op_left_frame.setObjectName("op_left_frame")
            self.left_pass_but = QtWidgets.QRadioButton(self.op_left_frame)
            self.left_pass_but.setEnabled(True)
            self.left_pass_but.setGeometry(QtCore.QRect(10, 10, 16, 16))
            self.left_pass_but.setFont(font)
            self.left_pass_but.setText("")
            self.left_pass_but.setObjectName("left_pass_but")
            self.left_pass_but.value = 'PASS'
            self.left_pass_but.name = 'left_strm'
            self.left_pass_but.toggled.connect(lambda: set_operator_test(self.left_pass_but))
            self.left_ntes_but = QtWidgets.QRadioButton(self.op_left_frame)
            self.left_ntes_but.setEnabled(True)
            self.left_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
            self.left_ntes_but.setFont(font)
            self.left_ntes_but.setText("")
            self.left_ntes_but.setChecked(True)
            self.left_ntes_but.setObjectName("left_ntes_but")
            self.left_ntes_but.value = ''
            self.left_ntes_but.name = 'left_strm'
            self.left_ntes_but.toggled.connect(lambda: set_operator_test(self.left_ntes_but))
            self.left_fail_but = QtWidgets.QRadioButton(self.op_left_frame)
            self.left_fail_but.setEnabled(True)
            self.left_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
            self.left_fail_but.setFont(font)
            self.left_fail_but.setText("")
            self.left_fail_but.setObjectName("left_fail_but")
            self.left_fail_but.value = 'FAIL'
            self.left_fail_but.name = 'left_strm'
            self.left_fail_but.toggled.connect(lambda: set_operator_test(self.left_fail_but))

            self.op_right_frame = QtWidgets.QFrame(self.operator_tests)
            self.op_right_frame.setGeometry(QtCore.QRect(160, 230, 131, 41))
            self.op_right_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.op_right_frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.op_right_frame.setLineWidth(0)
            self.op_right_frame.setObjectName("op_right_frame")
            self.right_pass_but = QtWidgets.QRadioButton(self.op_right_frame)
            self.right_pass_but.setEnabled(True)
            self.right_pass_but.setGeometry(QtCore.QRect(10, 10, 16, 16))
            self.right_pass_but.setFont(font)
            self.right_pass_but.setText("")
            self.right_pass_but.setObjectName("right_pass_but")
            self.right_pass_but.value = 'PASS'
            self.right_pass_but.name = 'right_strm'
            self.right_pass_but.toggled.connect(lambda: set_operator_test(self.right_pass_but))
            self.right_ntes_but = QtWidgets.QRadioButton(self.op_right_frame)
            self.right_ntes_but.setEnabled(True)
            self.right_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
            self.right_ntes_but.setFont(font)
            self.right_ntes_but.setText("")
            self.right_ntes_but.setChecked(True)
            self.right_ntes_but.setObjectName("right_ntes_but")
            self.right_ntes_but.value = ''
            self.right_ntes_but.name = 'right_strm'
            self.right_ntes_but.toggled.connect(lambda: set_operator_test(self.right_ntes_but))
            self.right_fail_but = QtWidgets.QRadioButton(self.op_right_frame)
            self.right_fail_but.setEnabled(True)
            self.right_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
            self.right_fail_but.setFont(font)
            self.right_fail_but.setText("")
            self.right_fail_but.setObjectName("right_fail_but")
            self.right_fail_but.value = 'FAIL'
            self.right_fail_but.name = 'right_strm'
            self.right_fail_but.toggled.connect(lambda: set_operator_test(self.right_fail_but))

            # if test_type == 'OAK-D-PRO' or test_type == 'OAK-D-PRO-POE':
            if 'oak-d pro' in eepromDataJson['productName'].lower():
                self.op_ir_frame = QtWidgets.QFrame(self.operator_tests)
                self.op_ir_frame.setGeometry(QtCore.QRect(160, 270, 131, 41))
                self.op_ir_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.op_ir_frame.setFrameShadow(QtWidgets.QFrame.Raised)
                self.op_ir_frame.setLineWidth(0)
                self.op_ir_frame.setObjectName("op_ir_frame")
                self.ir_pass_but = QtWidgets.QRadioButton(self.op_ir_frame)
                self.ir_pass_but.setEnabled(True)
                self.ir_pass_but.setGeometry(QtCore.QRect(10, 10, 16, 16))
                self.ir_pass_but.setFont(font)
                self.ir_pass_but.setText("")
                self.ir_pass_but.setObjectName("right_pass_but")
                self.ir_pass_but.value = 'PASS'
                self.ir_pass_but.name = 'ir_light'
                self.ir_pass_but.toggled.connect(lambda: set_operator_test(self.ir_pass_but))
                self.ir_ntes_but = QtWidgets.QRadioButton(self.op_ir_frame)
                self.ir_ntes_but.setEnabled(True)
                self.ir_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
                self.ir_ntes_but.setFont(font)
                self.ir_ntes_but.setText("")
                self.ir_ntes_but.setChecked(True)
                self.ir_ntes_but.setObjectName("right_ntes_but")
                self.ir_ntes_but.value = ''
                self.ir_ntes_but.name = 'ir_light'
                self.ir_ntes_but.toggled.connect(lambda: set_operator_test(self.ir_ntes_but))
                self.ir_fail_but = QtWidgets.QRadioButton(self.op_ir_frame)
                self.ir_fail_but.setEnabled(True)
                self.ir_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
                self.ir_fail_but.setFont(font)
                self.ir_fail_but.setText("")
                self.ir_fail_but.setObjectName("ir_fail_but")
                self.ir_fail_but.value = 'FAIL'
                self.ir_fail_but.name = 'ir_light'
                self.ir_fail_but.toggled.connect(lambda: set_operator_test(self.ir_fail_but))

        self.logs = QtWidgets.QGroupBox(self.centralwidget)
        self.logs.setGeometry(QtCore.QRect(10, 460, 741, 221))
        self.logs.setObjectName("logs")
        self.logs_title_label = QtWidgets.QLabel(self.logs)
        self.logs_title_label.setGeometry(QtCore.QRect(10, 20, 281, 21))
        self.logs_title_label.setObjectName("logs_title")
        self.logs_title_label.setText("Logs")
        self.date_time_label = QtWidgets.QLabel(self.logs)
        self.date_time_label.setGeometry(QtCore.QRect(10, 40, 281, 21))
        self.date_time_label.setObjectName("date_time_label")
        self.test_type_label = QtWidgets.QLabel(self.logs)
        self.test_type_label.setGeometry(QtCore.QRect(10, 60, 281, 21))
        self.test_type_label.setObjectName("test_type_label")
        self.prog_bar = QtWidgets.QProgressBar(self.logs)
        self.prog_bar.setGeometry(QtCore.QRect(540, 40, 118, 23))
        self.prog_bar.setProperty("value", 24)
        self.prog_bar.setObjectName("IMU_prog_bar")
        self.prog_bar.setMinimum(0)
        self.prog_bar.setMaximum(100)
        self.prog_bar.setValue(0)
        self.prog_label = QtWidgets.QLabel(self.logs)
        self.prog_label.setGeometry(QtCore.QRect(450, 40, 81, 17))
        self.prog_label.setObjectName("prog_label")
        self.logs_txt_browser = QtWidgets.QTextBrowser(self.logs)
        self.logs_txt_browser.setGeometry(QtCore.QRect(10, 90, 721, 121))
        self.logs_txt_browser.setObjectName("logs_txt_browser")
        UI_tests.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(UI_tests)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 766, 29))
        self.menubar.setObjectName("menubar")
        UI_tests.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(UI_tests)
        self.statusbar.setObjectName("statusbar")
        UI_tests.setStatusBar(self.statusbar)

        self.retranslateUi(UI_tests)
        QtCore.QMetaObject.connectSlotsByName(UI_tests)
        self.red_pallete = QtGui.QPalette()
        self.green_pallete = QtGui.QPalette()
        self.inactive_pallete = QtGui.QPalette()

        self.red_pallete.setColor(QtGui.QPalette.WindowText, QtCore.Qt.red)
        self.green_pallete.setColor(QtGui.QPalette.WindowText, QtCore.Qt.darkGreen)
        self.inactive_pallete.setColor(QtGui.QPalette.WindowText, QtCore.Qt.darkGray)
        # self.prew_out_rgb_res.setPalette(self.green_pallete)
        # self.save_but.clicked.connect(self.show_cameras)
        self.update_imu = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_result)
        self.timer.start(1000//FPS)

        # self.connect_timer = QtCore.QTimer()
        # self.connect_timer.connect(test_connexion)
        # self.timer.stop()

        self.threadpool = QThreadPool()
        self.scanning = False

    def retranslateUi(self, UI_tests):
        _translate = QtCore.QCoreApplication.translate
        UI_tests.setWindowTitle(_translate("UI_tests", "DepthAI UI Tests"))
        self.title.setText(_translate("UI_tests", f"<html><head/><body><p align=\"center\">Batch: {calib_path.parent.name} <br> Device: {calib_path.name}</p></body></html>"))
        self.connect_but.setText("CONNECT")
        self.connect_but.adjustSize()

        self.automated_tests.setTitle(_translate("UI_tests", "Automated Tests"))
        if test_type == 'OAK-1':
            self.automated_tests_labels.setText(_translate("UI_tests", OAK_ONE_LABELS))
        else:
            self.automated_tests_labels.setText(_translate("UI_tests", OAK_D_LABELS))
        self.operator_tests.setTitle(_translate("UI_tests", "Operator Tests"))
        self.NOT_TESTED_LABEL.setText(_translate("UI_tests", "<html><head/><body><p align=\"center\"><span style=\" font-size:11pt; color:#aaaa00;\">Not<br>Tested</span></p></body></html>"))
        self.FAIL_LABEL.setText(_translate("UI_tests", "<html><head/><body><p><span style=\" font-size:11pt; color:#ff0000;\">FAIL</span></p></body></html>"))
        self.operator_tests_label.setText(_translate("UI_tests", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"right\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">\n"
"JPEG Encoding <br><br>\n"
"preview-out-rgb <br><br>\n"
"Stream Left <br><br>\n"
"Stream Right <br><br>\n"
"IR Light</span></p></body></html>"))
        self.PASS_LABEL.setText(_translate("UI_tests", "<html><head/><body><p><span style=\" font-size:11pt; color:#00aa7f;\">PASS</span></p></body></html>"))
        self.logs.setTitle(_translate("UI_tests", ""))
        self.date_time_label.setText(_translate("UI_tests", "date_time: "))
        self.test_type_label.setText(_translate("UI_tests", "test_type: " + eepromDataJson['productName']))
        self.prog_label.setText(_translate("UI_tests", "Flash IMU"))
        # self.logs_txt_browser.setHtml(_translate("UI_tests", self.MB_INIT + "Test<br>" + "Test2<br>" + self.MB_END))
        self.print_logs(f'calib_path={calib_path}')

    def connect(self, f):
        class ConnectThread(QtCore.QThread):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def run(self):
                self.func()

        self.connectThread = ConnectThread(f)
        self.connectThread.start()

    @QtCore.pyqtSlot(str)
    def print_logs(self, new_log, log_level='INFO'):
        if new_log == 'clear':
            self.all_logs = ''
            return
        if log_level == 'ERROR':
            self.all_logs += '<p style="color:red">' + new_log + '</p>'
        elif log_level == 'WARNING':
            self.all_logs += '<p style="color:orange">' + new_log + '</p>'
        elif log_level == 'GREEN':
            self.all_logs += '<p style="color:green">' + new_log + '</p>'
        else:
            self.all_logs += new_log + '<br>'
        self.logs_txt_browser.setHtml(self.MB_INIT + self.all_logs + self.MB_END)
        self.logs_txt_browser.moveCursor(QtGui.QTextCursor.End)

    def update_prog_bar(self, value):
        self.prog_bar.setValue(int(value*100))

    def show_cameras(self):
        self.test_type_label.setText('test ' + eepromDataJson['productName'])
        if hasattr(self, 'depth_camera'):
            self.save_csv()
            clear_test_results()
            self.set_result()
            self.disconnect()
            self.rgb_ntes_but.setChecked(True)
            self.jpeg_ntes_but.setChecked(True)
            if test_type != 'OAK-1':
                self.right_ntes_but.setChecked(True)
                self.left_ntes_but.setChecked(True)
                # if test_type == 'OAK-D-PRO' or test_type == 'OAK-D-PRO-POE':
                if 'oak-d pro' in eepromDataJson['productName'].lower():
                    self.ir_ntes_but.setChecked(True)
            self.connect_but.setText("CONNECT")
            self.connect_but.adjustSize()
            return
        # self.print_logs('clear')
        if not self.scanning:
            self.connexion_result = Worker(test_connexion)
            self.connexion_result.signals.result.connect(self.connexion_slot)
            self.connexion_result.signals.finished.connect(self.end_conn)
            self.threadpool.start(self.connexion_result)
            self.scanning = True
            self.connect_but.setText('CANCEL')
            self.connect_but.adjustSize()
        else:
            self.connexion_result.stop()
            print('Canceling')
            self.connect_but.setText('CONNECT')
            self.connect_but.adjustSize()
            self.scanning = False
            self.threadpool.cancel(self.connexion_result)

    def end_conn(self):
        # print(s)
        print("end signal received")
        self.scanning = False

    def connexion_slot(self, signal):
        # self.threadpool.cancel(self.connexion_result)
        if not signal:
            self.print_logs('No camera detected, check the connexion and try again...', 'ERROR')
            return
        self.print_logs('Camera connected, starting tests...', 'GREEN')
        self.connect_but.setText("FLASHING")
        self.connect_but.adjustSize()
        self.connect_but.setChecked(False)
        self.connect_but.setCheckable(False)
        self.connect_but.setEnabled(False)
        # Update BL if PoE
        # if test_type == 'OAK-D-PRO-POE':
        if 'poe' in eepromDataJson['productName'].lower():
            self.update_bootloader()

        try:
            self.depth_camera = DepthAICamera()
            # Add IMU update cb

        except RuntimeError as ex:
            self.print_logs(f"Something went wrong, check connexion! - {ex}", log_level='ERROR')
            return
        # self.update_imu = True
        if imu_upgrade:
            self.depth_camera.device.setLogLevel(dai.LogLevel.INFO)
            self.depth_camera.device.addLogCallback(self.logMonitorImuFwUpdateCallback)
        else:
            self.connect_but.setChecked(False)
            self.connect_but.setCheckable(True)
            self.connect_but.setEnabled(True)
            self.connect_but.setText('DISCONNECT AND SAVE')
            self.connect_but.adjustSize()
            self.update_imu = True
        location = WIDTH, 0
        self.rgb = Camera(lambda: self.depth_camera.get_image('RGB'), colorMode, 'RGB Preview', location)
        self.rgb.show()
        location = WIDTH, prew_height + 80
        self.jpeg = Camera(lambda: self.depth_camera.get_image('JPEG'), colorMode, 'JPEG Preview', location)
        self.jpeg.show()
        if test_type != 'OAK-1':
            location = WIDTH + prew_width + 20, 0
            self.left = Camera(lambda: self.depth_camera.get_image('LEFT'), QtGui.QImage.Format_Grayscale8,
                               'LEFT Preview', location)
            self.left.show()
            location = WIDTH + prew_width + 20, prew_height + 80
            self.right = Camera(lambda: self.depth_camera.get_image('RIGHT'), QtGui.QImage.Format_Grayscale8,
                                'RIGHT Preview', location)
            self.right.show()
        self.print_logs('EEPROM backup saved at')
        self.print_logs(CALIB_BACKUP_FILE)
        eeprom_success, eeprom_msg, eeprom_data = self.depth_camera.flash_eeprom()
        if eeprom_success:
            self.print_logs('Flash EEPROM successful!', 'GREEN')
            test_result['eeprom_res'] = 'PASS'

            # Don't save full EEPROM json as it won't play well with regular csv.
            # Just save batchTime for now
            test_result['eeprom_data'] = str(eeprom_data['batchTime'])
            # test_result['eeprom_data'] = json.dumps(eeprom_data)
        else:
            self.print_logs(f'Flash EEPROM failed! - {eeprom_msg}', 'ERROR')
            test_result['eeprom_res'] = 'FAIL'
            test_result['eeprom_data'] = ''

    def set_result(self):
        global update_res
        if not self.update_imu:
            self.connect_but.setChecked(False)
            self.connect_but.setCheckable(True)
            self.connect_but.setEnabled(True)
            self.connect_but.setText('DISCONNECT AND SAVE')
            self.connect_but.adjustSize()
            self.update_imu = True

            # if test_type == 'OAK-D-PRO-POE':
            print(eepromDataJson['productName'].lower())
            if 'poe' in eepromDataJson['productName'].lower():
                if test_result['nor_flash_res'] == 'FAIL':
                    self.print_logs('BOOTLOADER UPDATE FAIL, RETRYING...')
                    global imu_upgrade
                    imu_upgrade = False
                    self.disconnect()
                    self.connexion_slot(True)
                    imu_upgrade = True
                    # self.update_bootloader()
                if test_result['nor_flash_res'] == 'FAIL':
                    self.print_logs('BOOTLOADER UPDATE FAIL!!!', log_level='ERROR')
                else:
                    self.print_logs('BOOTLOADER UPDATED SUCCESSFULLY', log_level='GREEN')


        time_string = datetime.now().strftime("%Y %m %d %H:%M:%S")
        self.date_time_label.setText('time: ' + time_string)
        if not update_res:
            return
        update_res = False
        if test_result['usb3_res'] == 'PASS':
            self.usb3_res.setPalette(self.green_pallete)
        elif test_result['usb3_res'] == 'SKIP':
            self.usb3_res.setPalette(self.inactive_pallete)
        else:
            self.usb3_res.setPalette(self.red_pallete)
        self.usb3_res.setText(test_result['usb3_res'])

        if test_result['eeprom_res'] == 'PASS':
            self.eeprom_res.setPalette(self.green_pallete)
        else:
            self.eeprom_res.setPalette(self.red_pallete)
        self.eeprom_res.setText(test_result['eeprom_res'])

        if test_result['rgb_cam_res'] == 'PASS':
            self.rgb_cam_res.setPalette(self.green_pallete)
        else:
            self.rgb_cam_res.setPalette(self.red_pallete)
        self.rgb_cam_res.setText(test_result['rgb_cam_res'])

        if test_result['jpeg_enc_res'] == 'PASS':
            self.jpeg_enc_res.setPalette(self.green_pallete)
        else:
            self.jpeg_enc_res.setPalette(self.red_pallete)
        self.jpeg_enc_res.setText(test_result['jpeg_enc_res'])

        if test_result['prew_out_rgb_res'] == 'PASS':
            self.prew_out_rgb_res.setPalette(self.green_pallete)
        else:
            self.prew_out_rgb_res.setPalette(self.red_pallete)
        self.prew_out_rgb_res.setText(test_result['prew_out_rgb_res'])

        if test_type != 'OAK-1':
            if test_result['left_cam_res'] == 'PASS':
                self.left_cam_res.setPalette(self.green_pallete)
            else:
                self.left_cam_res.setPalette(self.red_pallete)
            self.left_cam_res.setText(test_result['left_cam_res'])

            if test_result['right_cam_res'] == 'PASS':
                self.right_cam_res.setPalette(self.green_pallete)
            else:
                self.right_cam_res.setPalette(self.red_pallete)
            self.right_cam_res.setText(test_result['right_cam_res'])

            if test_result['left_strm_res'] == 'PASS':
                self.left_strm_res.setPalette(self.green_pallete)
            else:
                self.left_strm_res.setPalette(self.red_pallete)
            self.left_strm_res.setText(test_result['left_strm_res'])

            if test_result['right_strm_res'] == 'PASS':
                self.right_strm_res.setPalette(self.green_pallete)
            else:
                self.right_strm_res.setPalette(self.red_pallete)
            self.right_strm_res.setText(test_result['right_strm_res'])

    def update_bootloader_impl(self):
        self.print_logs('Check bootloader')
        deviceInfos = dai.DeviceBootloader.getAllAvailableDevices()
        if len(deviceInfos) <= 0:
            return (False, 'ERROR device was disconnected')

        try:
            with dai.DeviceBootloader(deviceInfos[0], allowFlashingBootloader=True) as bl:
                self.print_logs('Starting Update')
                self.prog_label.setText('Bootloader')
                # if test_type == 'OAK-D-PRO-POE':
                if 'poe' in eepromDataJson['productName'].lower():
                    self.print_logs('Flashing NETWORK bootloader...')
                    return bl.flashBootloader(dai.DeviceBootloader.Memory.FLASH, dai.DeviceBootloader.Type.NETWORK, self.update_prog_bar)
                else:
                    self.print_logs('Flashing USB bootloader...')
                    return bl.flashBootloader(dai.DeviceBootloader.Memory.FLASH, dai.DeviceBootloader.Type.USB, self.update_prog_bar)

        except RuntimeError as ex:
            # self.print_logs('Device communication failed, check connexions')
            return (False, f"Device communication failed, check connexions: {ex}")

    def update_bootloader(self):
        (result, message) = self.update_bootloader_impl()
        self.prog_label.setText('Flash IMU')
        self.update_prog_bar(0)
        if result:
            self.print_logs(f'Update bootloader: {message}')
            self.print_logs('Bootloader updated!', 'GREEN')
            test_result['nor_flash_res'] = 'PASS'
            return True
        else:
            self.print_logs(f'Failed to update bootloader: {message}', 'ERROR')
            test_result['nor_flash_res'] = 'FAIL'
            return False

    def logMonitorImuFwUpdateCallback(self, msg):
        # print('logMonitorIMuFwUpdateCallback')
        if 'IMU firmware update status' in msg.payload:
            try:
                percentage_float = float(msg.payload.split(":")[1].split("%")[0]) / 100.0
                self.prog_label.setText('Flash IMU')
                self.update_prog_bar(percentage_float)
            except Exception as ex:
                print(f'Could not parse fw update status: {ex}')
        if 'IMU firmware update succesful' in msg.payload:
            self.prog_label.setText('IMU PASS')
            self.update_prog_bar(1.0)
            self.print_logs_trigger.emit(f'Successfully updated IMU!')
            self.update_imu = False

        if 'IMU firmware update failed' in msg.payload:
            self.prog_label.setText('IMU FAIL')
            self.update_prog_bar(0.0)
            self.print_logs_trigger.emit(f'FAILED updating IMU!')
            self.update_imu = False
        # print(percentage_float)
        # print(percentage_float == 1, flush=True)
        # if percentage_float == 1:
        #     self.update_imu = False
        # print('END')


    def test_bootloader_version(self, version='0.0.15'):
        (result, info) = dai.DeviceBootloader.getFirstAvailableDevice()
        if not result:
            self.print_logs('ERROR device was disconnected!', 'ERROR')
            return False
        device = dai.DeviceBootloader(info)
        current_version = str(device.getVersion())
        # Skip version check for now
        # if current_version == version:
        #     self.print_logs('Bootloader up to date!')
        #     return True
        self.print_logs('Bootloader version is ' + current_version)
        self.print_logs('Starting bootloader update!')
        self.print_logs('Writing version ' + version + '...')
        (result, message) = self.update_bootloader()
        if result:
            self.print_logs('Bootloader updated!', 'GREEN')
            return True
        else:
            self.print_logs(f'Failed to update bootloader: {message}', 'ERROR')
            return False

    def save_csv(self):
        path = os.path.realpath(__file__).rsplit('/', 1)[0] + '/tests_result/' + eepromDataJson['productName'] + '.csv'
        print(path)
        if os.path.exists(path):
            file = open(path, 'a')
        else:
            file = open(path, 'w')
            if test_type in CSV_HEADER:
                file.write(CSV_HEADER[test_type] + '\n')
            else:
                file.write(CSV_HEADER['OAK-D'] + '\n')

        file.write(self.depth_camera.id)
        file.write(',' + eepromDataJson['productName'])
        file.write(',' + datetime.now().strftime("%Y %m %d %H:%M:%S"))

        if test_type in OAK_KEYS:
            auto_keys = OAK_KEYS[test_type]
        else:
            auto_keys = OAK_KEYS['OAK-D']

        if test_type in OP_OAK_KEYS:
            op_keys = OP_OAK_KEYS[test_type]
        else:
            op_keys = OP_OAK_KEYS['OAK-D']

        for key in auto_keys:
            if test_result[key] == '':
                file.write(',' + 'Not Tested')
            else:
                file.write(',' + test_result[key])
        for key in op_keys:
            if operator_tests[key] == '':
                file.write(',' + 'Not Tested')
            else:
                file.write(',' + operator_tests[key])
        file.write(',' + calib_path.parent.name)
        file.write(',' + calib_path.name)
        file.write('\n')
        file.close()
        self.print_logs('Test results for ' + eepromDataJson['productName'] + ' with id ' + self.depth_camera.id + ' had been saved!', 'GREEN')

    def close_event(self, event):
        self.disconnect()
        event.accept()
        sys.exit()

    def disconnect(self):
        if hasattr(self, 'depth_camera'):
            del self.rgb
            del self.jpeg
            if test_type != 'OAK-1':
                del self.left
                del self.right
            del self.depth_camera

    def batches_changed(self):
        print("Batches changed!")
        self.json_combo.clear()
        self.json_combo.addItems(self.jsons[self.batches_combo.currentText()])

def signal_handler(sig, frame):
    print('Closing app')
    ui.disconnect()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for test UI')
    parser.add_argument('-t', '--type', dest='camera_type', help='enter the type of device(OAK-1, OAK-D, OAK-D-PRO, OAK-D-LITE, OAK-D-PRO-POE)', default='OAK-D-PRO-POE')
    # parser.add_argument('-b', '--batch', dest='batch', help='enter the path to batch file', required=True)
    # CALIB_JSON_FILE = path = os.path.realpath(__file__).rsplit('/', 1)[0] + '/depthai_calib.json'
    CALIB_BACKUP_FILE = os.path.realpath(__file__).rsplit('/', 1)[0] + '/depthai_calib_backup.json'
    # parser.add_argument('--calib_json_file', '-c', dest='calib_json_file', help='Path to V6 calibration file in json', default=CALIB_JSON_FILE)
    args = parser.parse_args()
    test_type = args.camera_type.upper()
    # calib_path = args.calib_json_file
    # calib_path = args.batch
    # calib_path = None
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen()
    rect = screen.availableGeometry()
    prew_width = (rect.width() - WIDTH)//2 - 20
    prew_height = (rect.height())//2 - 80
    print(prew_width, prew_height)
    ui = UiTests()
    signal.signal(signal.SIGINT, signal_handler)
    ui.setupUi()
    UI_tests.show()
    sys.exit(app.exec_())

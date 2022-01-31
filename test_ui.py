from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QMessageBox
# from PyQt5.QtWidgets import QMessageBox
import numpy as np
import depthai as dai
import blobconverter

# self.usb3_res.setText('1')
# self.left_cam_res.setText('2')
# self.right_cam_res.setText('3')
# self.rgb_cam_res.setText('4')
# self.jpeg_enc_res.setText('5')
# self.prew_out_rgb_res.setText('6')
# self.left_strm_res.setText('7')
# self.right_strm_res.setText('8')

FPS = 10

test_result = {
    'usb3_res': '',
    'left_cam_res': '',
    'right_cam_res': '',
    'rgb_cam_res': '',
    'jpeg_enc_res': '',
    'prew_out_rgb_res': '',
    'left_strm_res': '',
    'right_strm_res': ''
}
update_res = False

class DepthAICamera():
    def __init__(self):
        global update_res
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.camLeft = self.pipeline.create(dai.node.MonoCamera)
        self.camRight = self.pipeline.create(dai.node.MonoCamera)

        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutLeft = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRight = self.pipeline.create(dai.node.XLinkOut)

        self.xoutRgb.setStreamName("rgb")
        self.xoutLeft.setStreamName("left")
        self.xoutRight.setStreamName("right")

        self.camRgb.setPreviewSize(640, 400)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        self.camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        self.camRgb.preview.link(self.xoutRgb.input)
        self.camLeft.out.link(self.xoutLeft.input)
        self.camRight.out.link(self.xoutRight.input)

        self.device = dai.Device(self.pipeline)

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

        print('Usb speed: ', self.device.getUsbSpeed().name)
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

    def start_queue(self):
        global update_res
        try:
            self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        except RuntimeError:
            test_result['prew_out_rgb_res'] = 'FAIL'
            update_res = True
        try:
            self.qLeft = self.device.getOutputQueue(name="left", maxSize=4, blocking=False)
        except RuntimeError:
            test_result['left_strm_res'] = 'FAIL'
            update_res = True
        try:
            self.qRight = self.device.getOutputQueue(name='right', maxSize=4, blocking=False)
        except RuntimeError:
            test_result['right_strm_res'] = 'FAIL'
            update_res = True

    def get_image(self, cam_type):
        global update_res
        try:
            image = None
            if cam_type == 'RGB':
                if test_result['rgb_cam_res'] == 'PASS':
                    in_rgb = self.qRgb.tryGet()
                    image = in_rgb.getCvFrame()
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

            return True, image
        except RuntimeError:
            if cam_type == 'RGB' and self._rgb_pass < self._NR_TEST_FRAMES:
                test_result['prew_out_rgb_res'] = 'FAIL'
                update_res = True
            if cam_type == 'LEFT' and self._left_pass < self._NR_TEST_FRAMES:
                test_result['left_strm_res'] == 'FAIL'
                update_res = True
            if cam_type == 'RIGHT' and self._right_pass < self._NR_TEST_FRAMES:
                test_result['right_cam_res'] == 'FAIL'
                update_res = True
            return False, None


class Camera(QtWidgets.QWidget):
    def __init__(self, get_image, camera_format):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.camera = QtWidgets.QLabel('ana are mere')
        layout.addWidget(self.camera)
        self.setLayout(layout)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(1000//FPS)
        self.get_image = get_image
        self.camera_format = camera_format

    def update_image(self):
        status, image = self.get_image()
        if status:
            q_image = QtGui.QImage(image.data, image.shape[1], image.shape[0], self.camera_format)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.camera.setPixmap(pixmap)
        else:
            # print('im hiding')
            self.hide()


class UiTests(object):
    def __init__(self):
        self.MB_INIT = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "</style></head><body style=\" font-family:\'Sans Serif\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
        self.MB_END = "</p></body></html>"
        self.all_logs = ""

    def setupUi(self, UI_tests):
        UI_tests.setObjectName("UI_tests")
        UI_tests.resize(766, 717)
        font = QtGui.QFont()
        font.setPointSize(13)
        UI_tests.setFont(font)
        self.centralwidget = QtWidgets.QWidget(UI_tests)
        self.centralwidget.setObjectName("centralwidget")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(10, 10, 751, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.save_but = QtWidgets.QPushButton(self.centralwidget)
        self.save_but.setGeometry(QtCore.QRect(510, 390, 61, 25))
        self.save_but.setObjectName("save_but")
        self.save_but.clicked.connect(self.show_cameras)
        self.automated_tests = QtWidgets.QGroupBox(self.centralwidget)
        self.automated_tests.setGeometry(QtCore.QRect(20, 70, 311, 341))
        self.automated_tests.setObjectName("automated_tests")
        self.automated_tests_labels = QtWidgets.QLabel(self.automated_tests)
        self.automated_tests_labels.setGeometry(QtCore.QRect(10, 30, 221, 301))
        self.automated_tests_labels.setObjectName("automated_tests_labels")
        self.right_cam_res = QtWidgets.QLabel(self.automated_tests)
        self.right_cam_res.setGeometry(QtCore.QRect(240, 110, 51, 31))
        self.right_cam_res.setObjectName("right_cam_res")
        self.prew_out_rgb_res = QtWidgets.QLabel(self.automated_tests)
        self.prew_out_rgb_res.setGeometry(QtCore.QRect(240, 220, 51, 41))
        self.prew_out_rgb_res.setObjectName("prew_out_rgb_res")
        self.jpeg_enc_res = QtWidgets.QLabel(self.automated_tests)
        self.jpeg_enc_res.setGeometry(QtCore.QRect(240, 190, 51, 31))
        self.jpeg_enc_res.setObjectName("jpeg_enc_res")
        self.left_cam_res = QtWidgets.QLabel(self.automated_tests)
        self.left_cam_res.setGeometry(QtCore.QRect(240, 70, 51, 31))
        self.left_cam_res.setObjectName("left_cam_res")
        self.right_strm_res = QtWidgets.QLabel(self.automated_tests)
        self.right_strm_res.setGeometry(QtCore.QRect(240, 300, 51, 31))
        self.right_strm_res.setObjectName("right_strm_res")
        self.usb3_res = QtWidgets.QLabel(self.automated_tests)
        self.usb3_res.setGeometry(QtCore.QRect(240, 30, 51, 21))
        self.usb3_res.setObjectName("usb3_res")
        self.rgb_cam_res = QtWidgets.QLabel(self.automated_tests)
        self.rgb_cam_res.setGeometry(QtCore.QRect(240, 150, 51, 31))
        self.rgb_cam_res.setObjectName("rgb_cam_res")
        self.left_strm_res = QtWidgets.QLabel(self.automated_tests)
        self.left_strm_res.setGeometry(QtCore.QRect(240, 260, 51, 41))
        self.left_strm_res.setObjectName("left_strm_res")
        self.operator_tests = QtWidgets.QGroupBox(self.centralwidget)
        self.operator_tests.setGeometry(QtCore.QRect(360, 70, 321, 291))
        self.operator_tests.setObjectName("operator_tests")
        self.NOT_TESTED_LABEL = QtWidgets.QLabel(self.operator_tests)
        self.NOT_TESTED_LABEL.setGeometry(QtCore.QRect(200, 30, 61, 61))
        self.NOT_TESTED_LABEL.setObjectName("NOT_TESTED_LABEL")
        self.op_rgb_frame = QtWidgets.QFrame(self.operator_tests)
        self.op_rgb_frame.setGeometry(QtCore.QRect(160, 140, 131, 41))
        self.op_rgb_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.op_rgb_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.op_rgb_frame.setLineWidth(0)
        self.op_rgb_frame.setObjectName("op_rgb_frame")
        self.rgb_fail_but = QtWidgets.QRadioButton(self.op_rgb_frame)
        self.rgb_fail_but.setEnabled(True)
        self.rgb_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.rgb_fail_but.setFont(font)
        self.rgb_fail_but.setText("")
        self.rgb_fail_but.setObjectName("rgb_fail_but")
        self.rgb_ntes_but = QtWidgets.QRadioButton(self.op_rgb_frame)
        self.rgb_ntes_but.setEnabled(True)
        self.rgb_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.rgb_ntes_but.setFont(font)
        self.rgb_ntes_but.setText("")
        self.rgb_ntes_but.setChecked(True)
        self.rgb_ntes_but.setObjectName("rgb_ntes_but")
        self.rgb_pass_but = QtWidgets.QRadioButton(self.op_rgb_frame)
        self.rgb_pass_but.setEnabled(True)
        self.rgb_pass_but.setGeometry(QtCore.QRect(10, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.rgb_pass_but.setFont(font)
        self.rgb_pass_but.setText("")
        self.rgb_pass_but.setObjectName("rgb_pass_but")
        self.FAIL_LABEL = QtWidgets.QLabel(self.operator_tests)
        self.FAIL_LABEL.setGeometry(QtCore.QRect(270, 50, 41, 21))
        self.FAIL_LABEL.setObjectName("FAIL_LABEL")
        self.op_right_frame = QtWidgets.QFrame(self.operator_tests)
        self.op_right_frame.setGeometry(QtCore.QRect(160, 230, 131, 41))
        self.op_right_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.op_right_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.op_right_frame.setLineWidth(0)
        self.op_right_frame.setObjectName("op_right_frame")
        self.right_fail_but = QtWidgets.QRadioButton(self.op_right_frame)
        self.right_fail_but.setEnabled(True)
        self.right_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.right_fail_but.setFont(font)
        self.right_fail_but.setText("")
        self.right_fail_but.setObjectName("right_fail_but")
        self.right_ntes_but = QtWidgets.QRadioButton(self.op_right_frame)
        self.right_ntes_but.setEnabled(True)
        self.right_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.right_ntes_but.setFont(font)
        self.right_ntes_but.setText("")
        self.right_ntes_but.setChecked(True)
        self.right_ntes_but.setObjectName("right_ntes_but")
        self.right_pass_but = QtWidgets.QRadioButton(self.op_right_frame)
        self.right_pass_but.setEnabled(True)
        self.right_pass_but.setGeometry(QtCore.QRect(10, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.right_pass_but.setFont(font)
        self.right_pass_but.setText("")
        self.right_pass_but.setObjectName("right_pass_but")
        self.op_jpeg_frame = QtWidgets.QFrame(self.operator_tests)
        self.op_jpeg_frame.setGeometry(QtCore.QRect(160, 90, 131, 41))
        self.op_jpeg_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.op_jpeg_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.op_jpeg_frame.setLineWidth(0)
        self.op_jpeg_frame.setObjectName("op_jpeg_frame")
        self.jpeg_fail_but = QtWidgets.QRadioButton(self.op_jpeg_frame)
        self.jpeg_fail_but.setEnabled(True)
        self.jpeg_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.jpeg_fail_but.setFont(font)
        self.jpeg_fail_but.setText("")
        self.jpeg_fail_but.setObjectName("jpeg_fail_but")
        self.jpeg_ntes_but = QtWidgets.QRadioButton(self.op_jpeg_frame)
        self.jpeg_ntes_but.setEnabled(True)
        self.jpeg_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.jpeg_ntes_but.setFont(font)
        self.jpeg_ntes_but.setText("")
        self.jpeg_ntes_but.setChecked(True)
        self.jpeg_ntes_but.setObjectName("jpeg_ntes_but")
        self.jpeg_pass_but = QtWidgets.QRadioButton(self.op_jpeg_frame)
        self.jpeg_pass_but.setEnabled(True)
        self.jpeg_pass_but.setGeometry(QtCore.QRect(10, 10, 14, 15))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.jpeg_pass_but.setFont(font)
        self.jpeg_pass_but.setText("")
        self.jpeg_pass_but.setObjectName("jpeg_pass_but")
        self.operator_tests_label = QtWidgets.QLabel(self.operator_tests)
        self.operator_tests_label.setGeometry(QtCore.QRect(10, 100, 131, 161))
        self.operator_tests_label.setObjectName("operator_tests_label")
        self.PASS_LABEL = QtWidgets.QLabel(self.operator_tests)
        self.PASS_LABEL.setGeometry(QtCore.QRect(150, 50, 41, 17))
        self.PASS_LABEL.setObjectName("PASS_LABEL")
        self.op_left_frame = QtWidgets.QFrame(self.operator_tests)
        self.op_left_frame.setGeometry(QtCore.QRect(160, 180, 131, 41))
        self.op_left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.op_left_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.op_left_frame.setLineWidth(0)
        self.op_left_frame.setObjectName("op_left_frame")
        self.left_fail_but = QtWidgets.QRadioButton(self.op_left_frame)
        self.left_fail_but.setEnabled(True)
        self.left_fail_but.setGeometry(QtCore.QRect(110, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.left_fail_but.setFont(font)
        self.left_fail_but.setText("")
        self.left_fail_but.setObjectName("left_fail_but")
        self.left_ntes_but = QtWidgets.QRadioButton(self.op_left_frame)
        self.left_ntes_but.setEnabled(True)
        self.left_ntes_but.setGeometry(QtCore.QRect(60, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.left_ntes_but.setFont(font)
        self.left_ntes_but.setText("")
        self.left_ntes_but.setChecked(True)
        self.left_ntes_but.setObjectName("left_ntes_but")
        self.left_pass_but = QtWidgets.QRadioButton(self.op_left_frame)
        self.left_pass_but.setEnabled(True)
        self.left_pass_but.setGeometry(QtCore.QRect(10, 10, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.left_pass_but.setFont(font)
        self.left_pass_but.setText("")
        self.left_pass_but.setObjectName("left_pass_but")
        self.logs = QtWidgets.QGroupBox(self.centralwidget)
        self.logs.setGeometry(QtCore.QRect(10, 430, 741, 221))
        self.logs.setObjectName("logs")
        self.date_time_label = QtWidgets.QLabel(self.logs)
        self.date_time_label.setGeometry(QtCore.QRect(10, 40, 281, 21))
        self.date_time_label.setObjectName("date_time_label")
        self.test_type_label = QtWidgets.QLabel(self.logs)
        self.test_type_label.setGeometry(QtCore.QRect(10, 60, 281, 21))
        self.test_type_label.setObjectName("test_type_label")
        self.IMU_prog_bar = QtWidgets.QProgressBar(self.logs)
        self.IMU_prog_bar.setGeometry(QtCore.QRect(540, 40, 118, 23))
        self.IMU_prog_bar.setProperty("value", 24)
        self.IMU_prog_bar.setObjectName("IMU_prog_bar")
        self.FLASH_IMU_LABEL = QtWidgets.QLabel(self.logs)
        self.FLASH_IMU_LABEL.setGeometry(QtCore.QRect(450, 40, 81, 17))
        self.FLASH_IMU_LABEL.setObjectName("FLASH_IMU_LABEL")
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

        self.red_pallete.setColor(QtGui.QPalette.WindowText, QtCore.Qt.red)
        self.green_pallete.setColor(QtGui.QPalette.WindowText, QtCore.Qt.darkGreen)
        # self.prew_out_rgb_res.setPalette(self.green_pallete)
        # self.save_but.clicked.connect(self.show_cameras)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_result)
        self.timer.start(1000//FPS)

    def retranslateUi(self, UI_tests):
        _translate = QtCore.QCoreApplication.translate
        UI_tests.setWindowTitle(_translate("UI_tests", "MainWindow"))
        self.title.setText(_translate("UI_tests", "<html><head/><body><p align=\"center\">UNIT TEST IN PROGRESS</p></body></html>"))
        self.save_but.setText("SAVE")
        self.automated_tests.setTitle(_translate("UI_tests", "Automated Tests"))
        self.automated_tests_labels.setText(_translate("UI_tests", "<html><head/><body><p align=\"right\"><span style=\" font-size:14pt;\">USB3</span></p><p align=\"right\"><span style=\" font-size:14pt;\">Left camera connected</span></p><p align=\"right\"><span style=\" font-size:14pt;\">Right Camera Connected</span></p><p align=\"right\"><span style=\" font-size:14pt;\">RGB Camera connected</span></p><p align=\"right\"><span style=\" font-size:14pt;\">JPEG Encoding Stream</span></p><p align=\"right\"><span style=\" font-size:14pt;\">preview-out-rgb Stream</span></p><p align=\"right\"><span style=\" font-size:14pt;\">left Stream</span></p><p align=\"right\"><span style=\" font-size:14pt;\">right Stream</span></p></body></html>"))
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
"Stream Right</span></p></body></html>"))
        self.PASS_LABEL.setText(_translate("UI_tests", "<html><head/><body><p><span style=\" font-size:11pt; color:#00aa7f;\">PASS</span></p></body></html>"))
        self.logs.setTitle(_translate("UI_tests", "Logs"))
        self.date_time_label.setText(_translate("UI_tests", "date_time: "))
        self.test_type_label.setText(_translate("UI_tests", "test_type: "))
        self.FLASH_IMU_LABEL.setText(_translate("UI_tests", "Flash IMU"))
        # self.logs_txt_browser.setHtml(_translate("UI_tests", self.MB_INIT + "Test<br>" + "Test2<br>" + self.MB_END))

    def print_logs(self, new_log):
        self.all_logs += new_log + '<br>'
        self.logs_txt_browser.setHtml(self.MB_INIT + self.all_logs + self.MB_END)
        self.logs_txt_browser.moveCursor(QtGui.QTextCursor.End)

    def test_connexion(self):
        (result, info) = dai.DeviceBootloader.getFirstAvailableDevice()
        if result:
            self.print_logs('TEST check if device connected: PASS')
            return True
        self.print_logs('TEST check if device connected: FAILED')
        return False

    def show_cameras(self):
        if hasattr(self, 'depth_camera'):
            self.print_logs('Camera already connected')
            self.rgb.show()
            self.left.show()
            self.right.show()
            return
        if self.test_connexion():
            self.print_logs('Camera connected, starting tests...')
            self.depth_camera = DepthAICamera()
            self.rgb = Camera(lambda: self.depth_camera.get_image('RGB'), QtGui.QImage.Format_BGR888)
            self.rgb.show()
            self.left = Camera(lambda: self.depth_camera.get_image('LEFT'), QtGui.QImage.Format_Grayscale8)
            self.left.show()
            self.right = Camera(lambda: self.depth_camera.get_image('RIGHT'), QtGui.QImage.Format_Grayscale8)
            self.right.show()
        else:
            print(locals())
            self.print_logs('No camera detected, check the connexion and try again...')

    def set_result(self):
        global update_res
        if not update_res:
            return
        update_res = False
        if test_result['usb3_res'] == 'PASS':
            self.usb3_res.setPalette(self.green_pallete)
        else:
            self.usb3_res.setPalette(self.red_pallete)
        self.usb3_res.setText(test_result['usb3_res'])

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

    # def update_bootloader(self):
    #     (result, device) = depthai.DeviceBootloader.getFirstAvailableDevice()
    #     if not result:
    #         self.print_logs('ERROR device was dissconected!')
    #         return False
    #     bootloader = depthai.DeviceBootloader(device, allowFlashingBootloader=True)
    #     # progress = lambda p: self.print_logs(f'Flashing progress: {p * 100:.1f}%')
    #     # bootloader.flashBootloader(progress)
    #     return True


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    UI_tests = QtWidgets.QMainWindow()
    ui = UiTests()
    ui.setupUi(UI_tests)
    UI_tests.show()
    ui.test_connexion()
    # ui.update_bootloader()
    sys.exit(app.exec_())


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibrate.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer,QDateTime
import cv2
# !/usr/bin/env python3
import argparse
import json
import shutil
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

import depthai_helpers.calibration_utils as calibUtils
import calibrate
from multiprocessing import Process, Pipe

# wrapper to run the camera in another process
def run_main_camera(options, show_img=cv2.imshow, waitKey=cv2.waitKey):
    main = calibrate.Main(options, show_img, waitKey)
    main.run()

class Ui_MainWindow(object):
    def __init__(self):
        self.camera_type = 'oak-d'
        self.options = {
            'count': 1,
            'squareSizeCm': 2.35,
            'markerSizeCm': 1.7625000000000002,
            'defaultBoard': True,
            'squaresX': 11,
            'squaresY': 8,
            'rectifiedDisp': True,
            'disableRgb': False,
            'swapLR': False,
            'mode': ['capture', 'process'],
            'board': 'bw1098obc',
            'invert_v': False,
            'invert_h': True,
            'maxEpiploarError': 1.0,
            'cameraMode': 'perspective',
            'rgbLensPosition': 135,
            'fps': 30}
        self.disply_width = 640
        self.display_height = 480
        self.key = 0
        self.key_out, self.key_in = Pipe()
        self.image_in, self.image_out = Pipe()

    def setup_ui(self, MainWindow):
        # Main Window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(937, 740)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Assets/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 923, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # Title
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(10, 10, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.title.setFont(font)
        self.title.setObjectName("title")
        # Select Model
        self.select_box = QtWidgets.QGroupBox(self.centralwidget)
        self.select_box.setGeometry(QtCore.QRect(10, 50, 261, 251))
        font.setPointSize(11)
        self.select_box.setFont(font)
        self.select_box.setObjectName("select_box")
        # OAK-1 option
        self.oak_1 = QtWidgets.QRadioButton(self.select_box)
        self.oak_1.setGeometry(QtCore.QRect(10, 40, 99, 21))
        self.oak_1.setFont(font)
        self.oak_1.setObjectName("oak_1")
        self.oak_1.value = 'oak-1'
        self.oak_1.toggled.connect(lambda: self.camera_select(self.oak_1))
        # OAK-D-Lite option
        self.oak_d_lite = QtWidgets.QRadioButton(self.select_box)
        self.oak_d_lite.setGeometry(QtCore.QRect(10, 70, 99, 21))
        self.oak_d_lite.setFont(font)
        self.oak_d_lite.setObjectName("oak_d_lite")
        self.oak_d_lite.value = 'oak-d-lite'
        self.oak_d_lite.toggled.connect(lambda: self.camera_select(self.oak_d_lite))
        # Oak-D option
        self.oak_d = QtWidgets.QRadioButton(self.select_box)
        self.oak_d.setGeometry(QtCore.QRect(10, 100, 99, 21))
        self.oak_d.setFont(font)
        self.oak_d.setChecked(True)
        self.oak_d.setObjectName("oak_d")
        self.oak_d.value = 'oak-d'
        self.oak_d.toggled.connect(lambda: self.camera_select(self.oak_d))
        # Oak-D-pro option
        self.oak_d_pro = QtWidgets.QRadioButton(self.select_box)
        self.oak_d_pro.setGeometry(QtCore.QRect(10, 130, 99, 21))
        self.oak_d_pro.setFont(font)
        self.oak_d_pro.setObjectName("oak_d_pro")
        self.oak_d_pro.value = 'oak-d-pro'
        self.oak_d_pro.toggled.connect(lambda: self.camera_select(self.oak_d_pro))
        # RaspberryPi Compute Module option
        self.rpi = QtWidgets.QRadioButton(self.select_box)
        self.rpi.setGeometry(QtCore.QRect(10, 160, 241, 21))
        self.rpi.setObjectName("rpi")
        self.rpi.value = 'rpi-module'
        self.rpi.toggled.connect(lambda: self.camera_select(self.rpi))
        # RaspberryPi Hat
        self.rpi_hat = QtWidgets.QRadioButton(self.select_box)
        self.rpi_hat.setGeometry(QtCore.QRect(10, 190, 151, 21))
        self.rpi_hat.setObjectName("radioButton_3")
        self.rpi_hat.value = 'rpi-hat'
        self.rpi_hat.toggled.connect(lambda: self.camera_select(self.rpi_hat))
        # USB3 with Modular Cameras
        self.usb3 = QtWidgets.QRadioButton(self.select_box)
        self.usb3.setGeometry(QtCore.QRect(10, 220, 241, 21))
        self.usb3.setObjectName("radioButton_2")
        self.usb3.value = 'usb3-module'
        self.usb3.toggled.connect(lambda: self.camera_select(self.usb3))
        # Buttons
        self.calibrate_but = QtWidgets.QPushButton(self.centralwidget)
        self.calibrate_but.setGeometry(QtCore.QRect(20, 370, 91, 31))
        self.calibrate_but.setFont(font)
        self.calibrate_but.setObjectName("calibrate_but")
        self.calibrate_but.clicked.connect(self.capture)
        self.connect_but = QtWidgets.QPushButton(self.centralwidget)
        self.connect_but.setGeometry(QtCore.QRect(130, 370, 81, 31))
        self.connect_but.setFont(font)
        self.connect_but.setObjectName("connect_but")
        self.connect_but.clicked.connect(self.start_calibration)
        # Square size
        self.square_size_in = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.square_size_in.setGeometry(QtCore.QRect(20, 330, 62, 24))
        self.square_size_in.setSingleStep(0.1)
        self.square_size_in.setProperty("value", 2.0)
        self.square_size_in.setObjectName("square_size_in")
        self.sq_size_label = QtWidgets.QLabel(self.centralwidget)
        self.sq_size_label.setGeometry(QtCore.QRect(20, 310, 131, 16))
        font.setPointSize(12)
        self.sq_size_label.setFont(font)
        self.sq_size_label.setObjectName("sq_size_label")
        # Charuco check
        self.charuco_check = QtWidgets.QCheckBox(self.centralwidget)
        self.charuco_check.setGeometry(QtCore.QRect(20, 410, 191, 21))
        font.setPointSize(11)
        self.charuco_check.setFont(font)
        self.charuco_check.setObjectName("charuco_check")
        # Mirror check
        self.mirror_check = QtWidgets.QCheckBox(self.centralwidget)
        self.mirror_check.setGeometry(QtCore.QRect(20, 440, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.mirror_check.setFont(font)
        self.mirror_check.setChecked(True)
        self.mirror_check.setObjectName("mirror_check")
        # Logs
        self.logs_box = QtWidgets.QGroupBox(self.centralwidget)
        self.logs_box.setGeometry(QtCore.QRect(20, 500, 901, 191))
        self.logs_box.setFont(font)
        self.logs_box.setObjectName("logs_box")
        self.logs = QtWidgets.QTextBrowser(self.logs_box)
        self.logs.setGeometry(QtCore.QRect(10, 30, 881, 151))
        self.logs.setObjectName("logs")
        MainWindow.setCentralWidget(self.centralwidget)
        # Image
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(280, 10, 640, 480))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("Assets/oak-d.jpg"))
        self.image.setObjectName("image")
        # Timer
        self.a = 0
        self.timer = QTimer(self.centralwidget)
        self.timer.timeout.connect(self.update_image)

        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Calibration"))
        self.title.setText(_translate("MainWindow", "DepthAI Calibration"))
        self.select_box.setTitle(_translate("MainWindow", "Select Model"))
        self.oak_1.setText(_translate("MainWindow", "OAK-1"))
        self.oak_d_lite.setText(_translate("MainWindow", "OAK-D-Lite"))
        self.oak_d.setText(_translate("MainWindow", "OAK-D"))
        self.oak_d_pro.setText(_translate("MainWindow", "OAK-D-Pro"))
        self.rpi.setText(_translate("MainWindow", "RaspberryPi Compute Module"))
        self.rpi_hat.setText(_translate("MainWindow", "RaspberryPi Hat"))
        self.usb3.setText(_translate("MainWindow", "USB3 with Modular Cameras"))
        self.connect_but.setText(_translate("MainWindow", "Connect"))
        self.calibrate_but.setText(_translate("MainWindow", "Calibrate"))
        self.sq_size_label.setText(_translate("MainWindow", "Square Size(cm)"))
        self.charuco_check.setText(_translate("MainWindow", "Show charuco board"))
        self.mirror_check.setText(_translate("MainWindow", "Mirror image"))
        self.logs_box.setTitle(_translate("MainWindow", "Logs"))

    def camera_select(self, model):
        if model.value == "oak-d":
            self.image.setPixmap(QtGui.QPixmap("Assets/oak-d.jpg"))
            self.camera_type = 'oak-d'
        elif model.value == "oak-d-lite":
            self.image.setPixmap(QtGui.QPixmap("Assets/oak-d-lite.jpg"))
            self.camera_type = 'oak-d-lite'
        elif model.value == "oak-d-pro":
            self.image.setPixmap(QtGui.QPixmap("Assets/oak-d-pro.jpg"))
            self.camera_type = 'oak-d-pro'
        elif model.value == "oak-1":
            self.image.setPixmap(QtGui.QPixmap("Assets/oak-1.jpg"))
            self.camera_type = 'oak-1'
        elif model.value == 'rpi-module':
            self.image.setPixmap(QtGui.QPixmap("Assets/rpi-compute-mod.jpg"))
            self.camera_type = 'rpi-module'
        elif model.value == 'rpi-hat':
            self.image.setPixmap(QtGui.QPixmap("Assets/rpi-hat.jpg"))
            self.camera_type = 'rpi-hat'
        elif model.value == 'usb3-module':
            self.image.setPixmap(QtGui.QPixmap("Assets/usb3.jpg"))
            self.camera_type = 'usb3-module'

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_image(self):
        result = self.image_in.poll(0.5/self.options['fps'])
        if result:
            print('image received!')
            image = self.image_in.recv()
            image = self.convert_cv_qt(image)
            self.image.setPixmap(image)

    def send_image(self, *args):
        pass

    def start_calibration(self):
        if not hasattr(self, 'p'):
            self.p = Process(target=run_main_camera, args=(self.options,self.image_out,self.key_in,))
            self.p.start()
            self.timer.start(1000//self.options['fps'])
        else:
            self.key_out.send('q')
            self.p.join()
            self.p.terminate()
            print("Process terminated")
            del self.p


    def capture(self):
        self.key_out.send(' ')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setup_ui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

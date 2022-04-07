import cv2
from multiprocessing import Process, Pipe
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
import time

import calibrate


# wrapper to run the camera in another process
def run_main_camera(options, show_img=cv2.imshow, wait_key=cv2.waitKey):
    main = calibrate.Main(options, show_img, wait_key)
    main.run()


class MyImage(QtWidgets.QWidget):
    def __init__(self, title='Charuco', location=(0, 0), prew_width=1280, prew_height=800):
        super().__init__()
        border = 52
        layout = QtWidgets.QVBoxLayout()
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon('Assets/logo.png'))
        width, height = location
        self.move(width, height)
        self.image = QtWidgets.QLabel('Camera')
        self.image.resize(prew_width - border, prew_height - border)
        layout.addWidget(self.image)
        self.setLayout(layout)
        self.pixmap = QtGui.QPixmap("Assets/charuco.png")
        self.pixmap = self.pixmap.scaled(prew_width-border, prew_height-border,
                                         QtCore.Qt.KeepAspectRatio)
        self.image.setPixmap(self.pixmap)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        border = 52
        print(self.geometry())
        self.pixmap = QtGui.QPixmap("Assets/charuco.png")
        self.pixmap = self.pixmap.scaled(self.frameGeometry().width()-border, self.frameGeometry().height()-border,
                                         QtCore.Qt.KeepAspectRatio)
        self.image.setPixmap(self.pixmap)
        self.image.resize(self.geometry().width()-border, self.geometry().height()-border)


class UiMainWindow(object):
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
        self.display_width = 640
        self.display_height = 480
        self.old_display_width = 640
        self.old_display_height = 480
        self.key = 0
        self.camera_pixmap = QtGui.QPixmap("Assets/oak-d.jpg")
        # self.setup_ui(self)

    def setup_ui(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(993, 833)
        font = QtGui.QFont()
        font.setPointSize(11)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(330, 10, 640, 481))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("../../.designer/backup/Assets/oak-d.jpg"))
        self.image.setObjectName("image")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(10, 10, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.logs_box = QtWidgets.QGroupBox(self.centralwidget)
        self.logs_box.setGeometry(QtCore.QRect(20, 590, 951, 191))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.logs_box.setFont(font)
        self.logs_box.setObjectName("logs_box")
        self.logs = QtWidgets.QTextBrowser(self.logs_box)
        self.logs.setGeometry(QtCore.QRect(10, 30, 931, 151))
        self.logs.setObjectName("logs")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 40, 301, 551))
        self.tabWidget.setObjectName("tabWidget")
        self.basic_tab = QtWidgets.QWidget()
        self.basic_tab.setObjectName("basic_tab")
        self.select_box = QtWidgets.QGroupBox(self.basic_tab)
        self.select_box.setGeometry(QtCore.QRect(0, 0, 291, 61))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.select_box.setFont(font)
        self.select_box.setObjectName("select_box")
        self.camera_box = QtWidgets.QComboBox(self.select_box)
        self.camera_box.setGeometry(QtCore.QRect(10, 30, 273, 26))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera_box.sizePolicy().hasHeightForWidth())
        self.camera_box.setSizePolicy(sizePolicy)
        self.camera_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.camera_box.setObjectName("camera_box")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/oak-1.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon1, "")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/oak-d-lite.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon2, "")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/oak-d.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon3, "")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/oak-d-pro.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon4, "")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/rpi-compute-mod.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon5, "")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/rpi-hat.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon6, "")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("../../.designer/backup/Assets/usb3.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon7, "")
        self.fps_label = QtWidgets.QLabel(self.basic_tab)
        self.fps_label.setGeometry(QtCore.QRect(130, 130, 31, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.fps_label.setFont(font)
        self.fps_label.setObjectName("fps_label")
        self.rect_check = QtWidgets.QCheckBox(self.basic_tab)
        self.rect_check.setGeometry(QtCore.QRect(20, 290, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.rect_check.setFont(font)
        self.rect_check.setObjectName("rect_check")
        self.fps_in = QtWidgets.QSpinBox(self.basic_tab)
        self.fps_in.setGeometry(QtCore.QRect(170, 120, 51, 31))
        self.fps_in.setMaximum(120)
        self.fps_in.setProperty("value", 30)
        self.fps_in.setObjectName("fps_in")
        self.mirror_check = QtWidgets.QCheckBox(self.basic_tab)
        self.mirror_check.setGeometry(QtCore.QRect(20, 230, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.mirror_check.setFont(font)
        self.mirror_check.setChecked(True)
        self.mirror_check.setObjectName("mirror_check")
        self.lens_pos_label = QtWidgets.QLabel(self.basic_tab)
        self.lens_pos_label.setGeometry(QtCore.QRect(20, 160, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lens_pos_label.setFont(font)
        self.lens_pos_label.setObjectName("lens_pos_label")
        self.square_size_in = QtWidgets.QDoubleSpinBox(self.basic_tab)
        self.square_size_in.setGeometry(QtCore.QRect(170, 80, 71, 31))
        self.square_size_in.setSingleStep(0.1)
        self.square_size_in.setProperty("value", 2.0)
        self.square_size_in.setObjectName("square_size_in")
        self.swap_check = QtWidgets.QCheckBox(self.basic_tab)
        self.swap_check.setGeometry(QtCore.QRect(20, 260, 211, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.swap_check.setFont(font)
        self.swap_check.setObjectName("swap_check")
        self.lens_pos_in = QtWidgets.QSpinBox(self.basic_tab)
        self.lens_pos_in.setGeometry(QtCore.QRect(170, 160, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lens_pos_in.setFont(font)
        self.lens_pos_in.setMaximum(300)
        self.lens_pos_in.setProperty("value", 135)
        self.lens_pos_in.setObjectName("lens_pos_in")
        self.sqr_size_label = QtWidgets.QLabel(self.basic_tab)
        self.sqr_size_label.setGeometry(QtCore.QRect(30, 80, 131, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.sqr_size_label.setFont(font)
        self.sqr_size_label.setObjectName("sqr_size_label")
        self.charuco_check = QtWidgets.QCheckBox(self.basic_tab)
        self.charuco_check.setGeometry(QtCore.QRect(20, 200, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.charuco_check.setFont(font)
        self.charuco_check.setObjectName("charuco_check")
        self.tabWidget.addTab(self.basic_tab, "")
        self.advanced_tab = QtWidgets.QWidget()
        self.advanced_tab.setObjectName("advanced_tab")
        self.help_but = QtWidgets.QPushButton(self.advanced_tab)
        self.help_but.setGeometry(QtCore.QRect(10, 10, 61, 23))
        self.help_but.setObjectName("help_but")
        self.spinBox = QtWidgets.QSpinBox(self.advanced_tab)
        self.spinBox.setGeometry(QtCore.QRect(10, 40, 43, 24))
        self.spinBox.setObjectName("spinBox")
        self.label = QtWidgets.QLabel(self.advanced_tab)
        self.label.setGeometry(QtCore.QRect(80, 40, 57, 15))
        self.label.setObjectName("label")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.advanced_tab)
        self.doubleSpinBox.setGeometry(QtCore.QRect(10, 70, 62, 24))
        self.doubleSpinBox.setMinimum(2.2)
        self.doubleSpinBox.setSingleStep(0.05)
        self.doubleSpinBox.setProperty("value", 2.35)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.label_2 = QtWidgets.QLabel(self.advanced_tab)
        self.label_2.setGeometry(QtCore.QRect(80, 70, 121, 16))
        self.label_2.setObjectName("label_2")
        self.checkBox = QtWidgets.QCheckBox(self.advanced_tab)
        self.checkBox.setGeometry(QtCore.QRect(80, 100, 191, 21))
        self.checkBox.setObjectName("checkBox")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.advanced_tab)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(10, 100, 62, 24))
        self.doubleSpinBox_2.setReadOnly(False)
        self.doubleSpinBox_2.setMinimum(1.65)
        self.doubleSpinBox_2.setSingleStep(0.05)
        self.doubleSpinBox_2.setProperty("value", 1.75)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.checkBox_2 = QtWidgets.QCheckBox(self.advanced_tab)
        self.checkBox_2.setGeometry(QtCore.QRect(10, 130, 131, 21))
        self.checkBox_2.setObjectName("checkBox_2")
        self.spinBox_2 = QtWidgets.QSpinBox(self.advanced_tab)
        self.spinBox_2.setGeometry(QtCore.QRect(10, 160, 43, 24))
        self.spinBox_2.setProperty("value", 11)
        self.spinBox_2.setObjectName("spinBox_2")
        self.label_3 = QtWidgets.QLabel(self.advanced_tab)
        self.label_3.setGeometry(QtCore.QRect(60, 160, 81, 16))
        self.label_3.setObjectName("label_3")
        self.spinBox_3 = QtWidgets.QSpinBox(self.advanced_tab)
        self.spinBox_3.setGeometry(QtCore.QRect(10, 190, 43, 24))
        self.spinBox_3.setProperty("value", 8)
        self.spinBox_3.setObjectName("spinBox_3")
        self.label_4 = QtWidgets.QLabel(self.advanced_tab)
        self.label_4.setGeometry(QtCore.QRect(60, 190, 81, 16))
        self.label_4.setObjectName("label_4")
        self.checkBox_3 = QtWidgets.QCheckBox(self.advanced_tab)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 220, 151, 21))
        self.checkBox_3.setChecked(True)
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.advanced_tab)
        self.checkBox_4.setGeometry(QtCore.QRect(10, 240, 121, 21))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.advanced_tab)
        self.checkBox_5.setGeometry(QtCore.QRect(10, 260, 91, 21))
        self.checkBox_5.setObjectName("checkBox_5")
        self.lineEdit = QtWidgets.QLineEdit(self.advanced_tab)
        self.lineEdit.setGeometry(QtCore.QRect(10, 280, 161, 23))
        self.lineEdit.setObjectName("lineEdit")
        self.label_5 = QtWidgets.QLabel(self.advanced_tab)
        self.label_5.setGeometry(QtCore.QRect(180, 280, 57, 15))
        self.label_5.setObjectName("label_5")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.advanced_tab)
        self.lineEdit_2.setGeometry(QtCore.QRect(10, 310, 161, 23))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_6 = QtWidgets.QLabel(self.advanced_tab)
        self.label_6.setGeometry(QtCore.QRect(180, 310, 57, 15))
        self.label_6.setObjectName("label_6")
        self.checkBox_6 = QtWidgets.QCheckBox(self.advanced_tab)
        self.checkBox_6.setGeometry(QtCore.QRect(10, 340, 141, 21))
        self.checkBox_6.setObjectName("checkBox_6")
        self.checkBox_7 = QtWidgets.QCheckBox(self.advanced_tab)
        self.checkBox_7.setGeometry(QtCore.QRect(10, 370, 151, 21))
        self.checkBox_7.setObjectName("checkBox_7")
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.advanced_tab)
        self.doubleSpinBox_3.setGeometry(QtCore.QRect(10, 390, 62, 24))
        self.doubleSpinBox_3.setSingleStep(0.1)
        self.doubleSpinBox_3.setProperty("value", 1.0)
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.label_7 = QtWidgets.QLabel(self.advanced_tab)
        self.label_7.setGeometry(QtCore.QRect(80, 390, 191, 16))
        self.label_7.setObjectName("label_7")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.advanced_tab)
        self.lineEdit_3.setGeometry(QtCore.QRect(10, 420, 101, 23))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_8 = QtWidgets.QLabel(self.advanced_tab)
        self.label_8.setGeometry(QtCore.QRect(120, 420, 111, 16))
        self.label_8.setObjectName("label_8")
        self.spinBox_4 = QtWidgets.QSpinBox(self.advanced_tab)
        self.spinBox_4.setGeometry(QtCore.QRect(10, 450, 51, 24))
        self.spinBox_4.setMaximum(300)
        self.spinBox_4.setProperty("value", 135)
        self.spinBox_4.setObjectName("spinBox_4")
        self.label_9 = QtWidgets.QLabel(self.advanced_tab)
        self.label_9.setGeometry(QtCore.QRect(70, 450, 141, 16))
        self.label_9.setObjectName("label_9")
        self.spinBox_5 = QtWidgets.QSpinBox(self.advanced_tab)
        self.spinBox_5.setGeometry(QtCore.QRect(10, 480, 43, 24))
        self.spinBox_5.setMaximum(120)
        self.spinBox_5.setProperty("value", 30)
        self.spinBox_5.setObjectName("spinBox_5")
        self.label_10 = QtWidgets.QLabel(self.advanced_tab)
        self.label_10.setGeometry(QtCore.QRect(60, 480, 41, 16))
        self.label_10.setObjectName("label_10")
        self.pushButton = QtWidgets.QPushButton(self.advanced_tab)
        self.pushButton.setGeometry(QtCore.QRect(80, 10, 80, 23))
        self.pushButton.setObjectName("pushButton")
        self.tabWidget.addTab(self.advanced_tab, "")
        self.calibrate_but = QtWidgets.QPushButton(self.centralwidget)
        self.calibrate_but.setGeometry(QtCore.QRect(430, 520, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.calibrate_but.setFont(font)
        self.calibrate_but.setObjectName("calibrate_but")
        self.connect_but = QtWidgets.QPushButton(self.centralwidget)
        self.connect_but.setGeometry(QtCore.QRect(330, 520, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.connect_but.setFont(font)
        self.connect_but.setObjectName("connect_but")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 993, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Calibration"))
        self.title.setText(_translate("MainWindow", "DepthAI Calibration"))
        self.logs_box.setTitle(_translate("MainWindow", "Logs"))
        self.select_box.setTitle(_translate("MainWindow", "Select Model"))
        self.camera_box.setCurrentText(_translate("MainWindow", "OAK-1"))
        self.camera_box.setItemText(0, _translate("MainWindow", "OAK-1"))
        self.camera_box.setItemText(1, _translate("MainWindow", "OAK-D-Lite"))
        self.camera_box.setItemText(2, _translate("MainWindow", "OAK-D"))
        self.camera_box.setItemText(3, _translate("MainWindow", "OAK-D-Pro"))
        self.camera_box.setItemText(4, _translate("MainWindow", "RaspberryPi Compute Module"))
        self.camera_box.setItemText(5, _translate("MainWindow", "RaspberryPi Hat"))
        self.camera_box.setItemText(6, _translate("MainWindow", "USB3 with Modular Cameras"))
        self.fps_label.setText(_translate("MainWindow", "FPS"))
        self.rect_check.setText(_translate("MainWindow", "Show Rectified images"))
        self.mirror_check.setText(_translate("MainWindow", "Mirror image"))
        self.lens_pos_label.setText(_translate("MainWindow", "RGB Lens Position"))
        self.swap_check.setText(_translate("MainWindow", "Swap Left Right Cameras"))
        self.sqr_size_label.setText(_translate("MainWindow", "Square Size(cm)"))
        self.charuco_check.setText(_translate("MainWindow", "Show charuco board"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.basic_tab), _translate("MainWindow", "Basic"))
        self.help_but.setText(_translate("MainWindow", "Help"))
        self.label.setText(_translate("MainWindow", "Count"))
        self.label_2.setText(_translate("MainWindow", "Square Size Cm"))
        self.checkBox.setText(_translate("MainWindow", "Markersize Cm"))
        self.checkBox_2.setText(_translate("MainWindow", "Default Board"))
        self.label_3.setText(_translate("MainWindow", "Squares X"))
        self.label_4.setText(_translate("MainWindow", "Squares Y"))
        self.checkBox_3.setText(_translate("MainWindow", "Rectified Display"))
        self.checkBox_4.setText(_translate("MainWindow", "Disable RGB"))
        self.checkBox_5.setText(_translate("MainWindow", "Swap LR"))
        self.lineEdit.setText(_translate("MainWindow", "[\'capture\', \'process\']"))
        self.label_5.setText(_translate("MainWindow", "Mode"))
        self.lineEdit_2.setText(_translate("MainWindow", "BW10980BC"))
        self.label_6.setText(_translate("MainWindow", "Board"))
        self.checkBox_6.setText(_translate("MainWindow", "Invert Vertical"))
        self.checkBox_7.setText(_translate("MainWindow", "Invert Horizontal"))
        self.label_7.setText(_translate("MainWindow", "maximum Epipolar Error"))
        self.lineEdit_3.setText(_translate("MainWindow", "perspective"))
        self.label_8.setText(_translate("MainWindow", "Camera Mode"))
        self.label_9.setText(_translate("MainWindow", "RGB Lens Position"))
        self.label_10.setText(_translate("MainWindow", "FPS"))
        self.pushButton.setText(_translate("MainWindow", "Reset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.advanced_tab), _translate("MainWindow", "Advanced"))
        self.calibrate_but.setText(_translate("MainWindow", "Calibrate"))
        self.connect_but.setText(_translate("MainWindow", "Connect"))

    def camera_select(self, model):
        if model.value == "oak-d":
            self.camera_pixmap = QtGui.QPixmap("Assets/oak-d.jpg")
            self.options['board'] = 'bw1098obc'
        elif model.value == "oak-d-lite":
            self.camera_pixmap = QtGui.QPixmap("Assets/oak-d-lite.jpg")
            self.options['board'] = 'OAK-D-LITE'
        elif model.value == "oak-d-pro":
            self.camera_pixmap = QtGui.QPixmap("Assets/oak-d-pro.jpg")
            self.options['board'] = 'OAK-D-PRO'
        elif model.value == "oak-1":
            self.camera_pixmap = QtGui.QPixmap("Assets/oak-1.jpg")
            self.camera_type = 'oak-1'
        elif model.value == 'rpi-module':
            self.camera_pixmap = QtGui.QPixmap("Assets/rpi-compute-mod.jpg")
            self.camera_type = 'rpi-module'
        elif model.value == 'rpi-hat':
            self.camera_pixmap = QtGui.QPixmap("Assets/rpi-hat.jpg")
            self.camera_type = 'rpi-hat'
        elif model.value == 'usb3-module':
            self.camera_pixmap = QtGui.QPixmap("Assets/usb3.jpg")
            self.camera_type = 'usb3-module'
        self.image.setPixmap(self.camera_pixmap)

    def show_charuco(self):
        if self.charuco_check.isChecked():
            if not hasattr(self, 'charuco'):
                self.charuco = MyImage()
            self.charuco.show()
        else:
            if hasattr(self, 'charuco'):
                self.charuco.hide()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_image(self):
        self.image.setScaledContents(True)
        result = self.image_in.poll(2 / self.options['fps'])
        if self.camera.is_alive():
            if result:
                if not hasattr(self, 'camera'):
                    self.image.setPixmap(self.camera_pixmap)
                    return
                if not self.connect_but.isEnabled():
                    self.connect_but.setEnabled(True)
                    self.connect_but.setText("Disconnect")
                capture = self.image_in.recv()
                capture = self.convert_cv_qt(capture)
                self.image.setPixmap(capture)
                if self.old_display_width != capture.width() or self.old_display_height != capture.height():
                    self.image.resize(capture.width(), capture.height())
                    self.old_display_width = capture.width()
                    self.old_display_height = capture.height()
        else:
            if not self.connect_but.isEnabled():
                self.connect_but.setEnabled(True)

    def resize_call(self, a0: QtGui.QResizeEvent) -> None:
        border = 52
        self.logs_box.setGeometry(QtCore.QRect(20, self.mainWindow.geometry().height()-border-191, 901, 191))
        self.display_width = self.mainWindow.geometry().width() - border - 280
        self.display_height = self.mainWindow.geometry().height() - border - 191
        if not hasattr(self, 'camera'):
            self.camera_pixmap = self.camera_pixmap.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
            self.image.resize(self.camera_pixmap.size().width(), self.camera_pixmap.size().height())

    def camera_connect(self):
        self.connect_but.setDisabled(True)
        self.mainWindow.repaint()
        if not hasattr(self, 'camera'):
            self.options['squareSizeCm'] = self.square_size_in.value()
            self.options['invert_h'] = self.mirror_check.isChecked()
            self.key_out, self.key_in = Pipe()
            self.image_in, self.image_out = Pipe()
            self.camera = Process(target=run_main_camera, args=(self.options, self.image_out, self.key_in,))
            self.camera.start()
            self.timer.start(1000 // self.options['fps'])
        else:
            self.connect_but.setDisabled(True)
            time.sleep(0.5)
            self.camera.terminate()
            self.image_in.close()
            self.image_out.close()
            self.key_out.close()
            self.key_in.close()
            del self.image_in
            del self.image_out
            del self.key_out
            del self.key_in
            self.camera.join()
            self.camera.close()
            del self.camera
            self.image.setPixmap(self.camera_pixmap)
            self.connect_but.setText("Connect")
            self.connect_but.setEnabled(True)
            self.timer.stop()

    def capture(self):
        self.key_out.send(' ')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    ui.setup_ui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

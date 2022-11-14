from PyQt5 import QtCore, QtGui, QtWidgets

import calibrate2 as calibrate
import cv2
from multiprocessing import Process, Pipe
import time
from pathlib import Path
import glob
import os
import depthai as dai


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(993, 721)
        font = QtGui.QFont()
        font.setPointSize(11)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./../.designer/backup/Assets/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(330, 70, 640, 481))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("./../.designer/backup/Assets/oak-d.jpg"))
        self.image.setObjectName("image")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(20, 10, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(20, 50, 301, 621))
        self.tabWidget.setObjectName("tabWidget")
        self.basic_tab = QtWidgets.QWidget()
        self.basic_tab.setObjectName("basic_tab")
        self.select_box = QtWidgets.QGroupBox(self.basic_tab)
        self.select_box.setGeometry(QtCore.QRect(0, 0, 291, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.select_box.setFont(font)
        self.select_box.setObjectName("select_box")
        self.camera_box = QtWidgets.QComboBox(self.select_box)
        self.camera_box.setGeometry(QtCore.QRect(10, 30, 94, 37))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera_box.sizePolicy().hasHeightForWidth())
        self.camera_box.setSizePolicy(sizePolicy)
        self.camera_box.setCurrentText("")
        self.camera_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.camera_box.setObjectName("camera_box")
        self.rect_check = QtWidgets.QCheckBox(self.basic_tab)
        self.rect_check.setGeometry(QtCore.QRect(20, 280, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.rect_check.setFont(font)
        self.rect_check.setObjectName("rect_check")
        self.mirror_check = QtWidgets.QCheckBox(self.basic_tab)
        self.mirror_check.setGeometry(QtCore.QRect(20, 220, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.mirror_check.setFont(font)
        self.mirror_check.setChecked(True)
        self.mirror_check.setObjectName("mirror_check")
        self.square_size_in = QtWidgets.QDoubleSpinBox(self.basic_tab)
        self.square_size_in.setGeometry(QtCore.QRect(160, 100, 71, 31))
        self.square_size_in.setSingleStep(0.1)
        self.square_size_in.setProperty("value", 2.0)
        self.square_size_in.setObjectName("square_size_in")
        self.swap_check = QtWidgets.QCheckBox(self.basic_tab)
        self.swap_check.setGeometry(QtCore.QRect(20, 250, 211, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.swap_check.setFont(font)
        self.swap_check.setObjectName("swap_check")
        self.sqr_size_label = QtWidgets.QLabel(self.basic_tab)
        self.sqr_size_label.setGeometry(QtCore.QRect(20, 100, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.sqr_size_label.setFont(font)
        self.sqr_size_label.setObjectName("sqr_size_label")
        self.charuco_check = QtWidgets.QCheckBox(self.basic_tab)
        self.charuco_check.setGeometry(QtCore.QRect(20, 190, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.charuco_check.setFont(font)
        self.charuco_check.setObjectName("charuco_check")
        self.marker_size_in = QtWidgets.QDoubleSpinBox(self.basic_tab)
        self.marker_size_in.setGeometry(QtCore.QRect(160, 140, 71, 31))
        self.marker_size_in.setSingleStep(0.1)
        self.marker_size_in.setProperty("value", 1.75)
        self.marker_size_in.setObjectName("marker_size_in")
        self.sqr_size_label_2 = QtWidgets.QLabel(self.basic_tab)
        self.sqr_size_label_2.setGeometry(QtCore.QRect(20, 140, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.sqr_size_label_2.setFont(font)
        self.sqr_size_label_2.setObjectName("sqr_size_label_2")
        self.slider_focus = QtWidgets.QSlider(self.basic_tab)
        self.slider_focus.setGeometry(QtCore.QRect(20, 370, 251, 31))
        self.slider_focus.setMaximum(255)
        self.slider_focus.setProperty("value", 135)
        self.slider_focus.setOrientation(QtCore.Qt.Horizontal)
        self.slider_focus.setObjectName("slider_focus")
        self.label = QtWidgets.QLabel(self.basic_tab)
        self.label.setGeometry(QtCore.QRect(20, 340, 67, 22))
        self.label.setObjectName("label")
        self.tabWidget.addTab(self.basic_tab, "")
        self.advanced_tab = QtWidgets.QWidget()
        self.advanced_tab.setObjectName("advanced_tab")
        self.help_but = QtWidgets.QPushButton(self.advanced_tab)
        self.help_but.setGeometry(QtCore.QRect(10, 10, 61, 23))
        self.help_but.setObjectName("help_but")
        self.count_spin = QtWidgets.QSpinBox(self.advanced_tab)
        self.count_spin.setGeometry(QtCore.QRect(10, 40, 43, 24))
        self.count_spin.setMinimum(1)
        self.count_spin.setObjectName("count_spin")
        self.l_count = QtWidgets.QLabel(self.advanced_tab)
        self.l_count.setGeometry(QtCore.QRect(80, 40, 57, 15))
        self.l_count.setObjectName("l_count")
        self.spin_sqr_size = QtWidgets.QDoubleSpinBox(self.advanced_tab)
        self.spin_sqr_size.setGeometry(QtCore.QRect(10, 70, 62, 24))
        self.spin_sqr_size.setMinimum(2.0)
        self.spin_sqr_size.setSingleStep(0.05)
        self.spin_sqr_size.setProperty("value", 2.0)
        self.spin_sqr_size.setObjectName("spin_sqr_size")
        self.l_sqr_size_cm = QtWidgets.QLabel(self.advanced_tab)
        self.l_sqr_size_cm.setGeometry(QtCore.QRect(80, 70, 121, 16))
        self.l_sqr_size_cm.setObjectName("l_sqr_size_cm")
        self.spin_markersize = QtWidgets.QDoubleSpinBox(self.advanced_tab)
        self.spin_markersize.setGeometry(QtCore.QRect(10, 100, 62, 24))
        self.spin_markersize.setReadOnly(False)
        self.spin_markersize.setMinimum(1.5)
        self.spin_markersize.setSingleStep(0.05)
        self.spin_markersize.setProperty("value", 1.52)
        self.spin_markersize.setObjectName("spin_markersize")
        self.check_def_board = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_def_board.setGeometry(QtCore.QRect(10, 130, 131, 21))
        self.check_def_board.setObjectName("check_def_board")
        self.spin_sqr_x = QtWidgets.QSpinBox(self.advanced_tab)
        self.spin_sqr_x.setGeometry(QtCore.QRect(10, 160, 43, 24))
        self.spin_sqr_x.setProperty("value", 11)
        self.spin_sqr_x.setObjectName("spin_sqr_x")
        self.l_squares_x = QtWidgets.QLabel(self.advanced_tab)
        self.l_squares_x.setGeometry(QtCore.QRect(60, 160, 81, 16))
        self.l_squares_x.setObjectName("l_squares_x")
        self.spin_sqr_y = QtWidgets.QSpinBox(self.advanced_tab)
        self.spin_sqr_y.setGeometry(QtCore.QRect(10, 190, 43, 24))
        self.spin_sqr_y.setProperty("value", 8)
        self.spin_sqr_y.setObjectName("spin_sqr_y")
        self.l_squares_y = QtWidgets.QLabel(self.advanced_tab)
        self.l_squares_y.setGeometry(QtCore.QRect(60, 190, 81, 16))
        self.l_squares_y.setObjectName("l_squares_y")
        self.check_rect_disp = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_rect_disp.setGeometry(QtCore.QRect(10, 220, 151, 21))
        self.check_rect_disp.setChecked(True)
        self.check_rect_disp.setObjectName("check_rect_disp")
        self.check_dis_rgb = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_dis_rgb.setGeometry(QtCore.QRect(10, 240, 121, 21))
        self.check_dis_rgb.setObjectName("check_dis_rgb")
        self.check_swapLR = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_swapLR.setGeometry(QtCore.QRect(10, 260, 91, 21))
        self.check_swapLR.setObjectName("check_swapLR")
        self.input_mode = QtWidgets.QLineEdit(self.advanced_tab)
        self.input_mode.setGeometry(QtCore.QRect(10, 280, 161, 23))
        self.input_mode.setObjectName("input_mode")
        self.l_mode = QtWidgets.QLabel(self.advanced_tab)
        self.l_mode.setGeometry(QtCore.QRect(180, 280, 57, 15))
        self.l_mode.setObjectName("l_mode")
        self.input_board = QtWidgets.QLineEdit(self.advanced_tab)
        self.input_board.setGeometry(QtCore.QRect(10, 310, 161, 23))
        self.input_board.setObjectName("input_board")
        self.l_board = QtWidgets.QLabel(self.advanced_tab)
        self.l_board.setGeometry(QtCore.QRect(180, 310, 57, 15))
        self.l_board.setObjectName("l_board")
        self.check_inv_vert = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_inv_vert.setGeometry(QtCore.QRect(10, 340, 141, 21))
        self.check_inv_vert.setObjectName("check_inv_vert")
        self.check_invert_horz = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_invert_horz.setGeometry(QtCore.QRect(10, 370, 151, 21))
        self.check_invert_horz.setObjectName("check_invert_horz")
        self.input_camera_mode = QtWidgets.QLineEdit(self.advanced_tab)
        self.input_camera_mode.setGeometry(QtCore.QRect(10, 400, 101, 23))
        self.input_camera_mode.setObjectName("input_camera_mode")
        self.label_8 = QtWidgets.QLabel(self.advanced_tab)
        self.label_8.setGeometry(QtCore.QRect(120, 400, 111, 16))
        self.label_8.setObjectName("label_8")
        self.spin_rgb_lens_pos = QtWidgets.QSpinBox(self.advanced_tab)
        self.spin_rgb_lens_pos.setGeometry(QtCore.QRect(10, 430, 51, 24))
        self.spin_rgb_lens_pos.setMaximum(300)
        self.spin_rgb_lens_pos.setProperty("value", 135)
        self.spin_rgb_lens_pos.setObjectName("spin_rgb_lens_pos")
        self.label_9 = QtWidgets.QLabel(self.advanced_tab)
        self.label_9.setGeometry(QtCore.QRect(70, 430, 141, 16))
        self.label_9.setObjectName("label_9")
        self.fps_in = QtWidgets.QSpinBox(self.advanced_tab)
        self.fps_in.setGeometry(QtCore.QRect(10, 460, 43, 24))
        self.fps_in.setMaximum(120)
        self.fps_in.setProperty("value", 30)
        self.fps_in.setObjectName("fps_in")
        self.l_fps = QtWidgets.QLabel(self.advanced_tab)
        self.l_fps.setGeometry(QtCore.QRect(60, 460, 41, 16))
        self.l_fps.setObjectName("l_fps")
        self.pushButton = QtWidgets.QPushButton(self.advanced_tab)
        self.pushButton.setGeometry(QtCore.QRect(80, 10, 80, 23))
        self.pushButton.setObjectName("pushButton")
        self.l_markersize = QtWidgets.QLabel(self.advanced_tab)
        self.l_markersize.setGeometry(QtCore.QRect(80, 100, 121, 16))
        self.l_markersize.setObjectName("l_markersize")
        self.spin_capture_delay = QtWidgets.QDoubleSpinBox(self.advanced_tab)
        self.spin_capture_delay.setGeometry(QtCore.QRect(10, 490, 62, 24))
        self.spin_capture_delay.setMinimum(0.0)
        self.spin_capture_delay.setSingleStep(0.05)
        self.spin_capture_delay.setProperty("value", 5.0)
        self.spin_capture_delay.setObjectName("spin_capture_delay")
        self.l_capture_delay = QtWidgets.QLabel(self.advanced_tab)
        self.l_capture_delay.setGeometry(QtCore.QRect(80, 490, 121, 21))
        self.l_capture_delay.setObjectName("l_capture_delay")
        self.check_debug = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_debug.setGeometry(QtCore.QRect(10, 520, 81, 31))
        self.check_debug.setObjectName("check_debug")
        self.check_factory_calib = QtWidgets.QCheckBox(self.advanced_tab)
        self.check_factory_calib.setGeometry(QtCore.QRect(10, 550, 171, 31))
        self.check_factory_calib.setObjectName("check_factory_calib")
        self.tabWidget.addTab(self.advanced_tab, "")
        self.calibrate_but = QtWidgets.QPushButton(self.centralwidget)
        self.calibrate_but.setGeometry(QtCore.QRect(430, 630, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.calibrate_but.setFont(font)
        self.calibrate_but.setObjectName("calibrate_but")
        self.connect_but = QtWidgets.QPushButton(self.centralwidget)
        self.connect_but.setGeometry(QtCore.QRect(340, 630, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.connect_but.setFont(font)
        self.connect_but.setObjectName("connect_but")
        self.delete_but = QtWidgets.QPushButton(self.centralwidget)
        self.delete_but.setGeometry(QtCore.QRect(530, 630, 91, 31))
        self.delete_but.setObjectName("delete_but")
        self.process_but = QtWidgets.QPushButton(self.centralwidget)
        self.process_but.setGeometry(QtCore.QRect(630, 630, 91, 31))
        self.process_but.setObjectName("process_but")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Calibration"))
        self.title.setText(_translate("MainWindow", "DepthAI Calibration"))
        self.select_box.setTitle(_translate("MainWindow", "Select Model"))
        self.rect_check.setText(_translate("MainWindow", "Show Rectified images"))
        self.mirror_check.setText(_translate("MainWindow", "Mirror image"))
        self.swap_check.setText(_translate("MainWindow", "Swap Left Right Cameras"))
        self.sqr_size_label.setText(_translate("MainWindow", "Square Size(cm)"))
        self.charuco_check.setText(_translate("MainWindow", "Show charuco board"))
        self.sqr_size_label_2.setText(_translate("MainWindow", "Marker Size(cm)"))
        self.label.setText(_translate("MainWindow", "Focus"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.basic_tab), _translate("MainWindow", "Basic"))
        self.help_but.setText(_translate("MainWindow", "Help"))
        self.l_count.setText(_translate("MainWindow", "Count"))
        self.l_sqr_size_cm.setText(_translate("MainWindow", "Square Size Cm"))
        self.check_def_board.setText(_translate("MainWindow", "Default Board"))
        self.l_squares_x.setText(_translate("MainWindow", "Squares X"))
        self.l_squares_y.setText(_translate("MainWindow", "Squares Y"))
        self.check_rect_disp.setText(_translate("MainWindow", "Rectified Display"))
        self.check_dis_rgb.setText(_translate("MainWindow", "Disable RGB"))
        self.check_swapLR.setText(_translate("MainWindow", "Swap LR"))
        self.input_mode.setText(_translate("MainWindow", "capture process"))
        self.l_mode.setText(_translate("MainWindow", "Mode"))
        self.input_board.setText(_translate("MainWindow", "BW10980BC"))
        self.l_board.setText(_translate("MainWindow", "Board"))
        self.check_inv_vert.setText(_translate("MainWindow", "Invert Vertical"))
        self.check_invert_horz.setText(_translate("MainWindow", "Invert Horizontal"))
        self.input_camera_mode.setText(_translate("MainWindow", "perspective"))
        self.label_8.setText(_translate("MainWindow", "Camera Mode"))
        self.label_9.setText(_translate("MainWindow", "RGB Lens Position"))
        self.l_fps.setText(_translate("MainWindow", "FPS"))
        self.pushButton.setText(_translate("MainWindow", "Reset"))
        self.l_markersize.setText(_translate("MainWindow", "Markersize Cm"))
        self.l_capture_delay.setText(_translate("MainWindow", "Capture Delay"))
        self.check_debug.setText(_translate("MainWindow", "Debug"))
        self.check_factory_calib.setText(_translate("MainWindow", "Factory Calibration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.advanced_tab), _translate("MainWindow", "Advanced"))
        self.calibrate_but.setText(_translate("MainWindow", "Calibrate"))
        self.connect_but.setText(_translate("MainWindow", "Connect"))
        self.delete_but.setText(_translate("MainWindow", "Delete"))
        self.process_but.setText(_translate("MainWindow", "Process"))


BOARDS_DIR = Path(__file__).resolve().parent / 'resources' / 'boards'

# wrapper to run the camera in another process
def run_main_camera(options, mxid, show_img=cv2.imshow, wait_key=cv2.waitKey):
    main = calibrate.Main(options, show_img, wait_key, mxid)
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


class Application(QtWidgets.QMainWindow):
    output_scale_factor = 0.5
    polygons = None
    width = None
    height = None
    current_polygon = 0
    images_captured_polygon = 0
    images_captured = 0

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.args = self.options
        self.display_width = 640
        self.display_height = 480
        self.old_display_width = 640
        self.old_display_height = 480
        self.key = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_image)
        # self.device_test = depthai.Device()
        self.ui.calibrate_but.clicked.connect(self.capture)
        # Get all devices
        devices = glob.glob(f'{BOARDS_DIR}/*.json')
        for dev in devices:
            board = os.path.basename(dev)[:-5]
            self.ui.camera_box.addItem(board)
            # with open(dev, 'r')
        self.ui.camera_box.activated.connect(lambda: self.change_event('board', self.ui.camera_box.currentText))

        self.ui.square_size_in.valueChanged.connect(lambda: self.change_event('squareSizeCm', self.ui.square_size_in.value))
        self.ui.fps_in.valueChanged.connect(lambda: self.change_event('fps', self.ui.fps_in.value))
        self.ui.marker_size_in.valueChanged.connect(lambda: self.change_event('markerSizeCm', self.ui.marker_size_in.value))
        self.ui.count_spin.valueChanged.connect(lambda : self.change_event('count', self.ui.count_spin.value))
        self.ui.spin_sqr_x.valueChanged.connect(lambda : self.change_event('squaresX', self.ui.spin_sqr_x.value))
        self.ui.spin_sqr_y.valueChanged.connect(lambda: self.change_event('squaresY', self.ui.spin_sqr_y.value))
        self.ui.spin_rgb_lens_pos.valueChanged.connect(lambda: self.change_event('rgbLensPosition', self.ui.spin_rgb_lens_pos.value))
        self.ui.spin_capture_delay.valueChanged.connect(lambda: self.change_event('captureDelay', self.ui.spin_capture_delay.value))

        self.ui.check_def_board.toggled.connect(lambda : self.change_event('defaultBoard', self.ui.check_def_board.isChecked))
        self.ui.check_rect_disp.toggled.connect(lambda: self.change_event('rectifiedDisp', self.ui.check_rect_disp.isChecked))
        self.ui.check_dis_rgb.toggled.connect(lambda: self.change_event('disableRgb', self.ui.check_dis_rgb.isChecked))
        self.ui.check_swapLR.toggled.connect(lambda: self.change_event('swapLR', self.ui.check_swapLR.isChecked))
        self.ui.check_inv_vert.toggled.connect(lambda: self.change_event('invert_v', self.ui.check_inv_vert.isChecked))
        self.ui.check_invert_horz.toggled.connect(lambda: self.change_event('invert_h', self.ui.check_invert_horz.isChecked))
        self.ui.check_debug.toggled.connect(lambda: self.change_event('debug', self.ui.check_debug.isChecked))
        self.ui.check_factory_calib.toggled.connect(lambda: self.change_event('factoryCalibration', self.ui.check_factory_calib.isChecked))

        self.ui.input_mode.textChanged.connect(lambda: self.change_event('mode', self.ui.input_mode.text().split))
        self.ui.input_board.textChanged.connect(lambda: self.change_event('board', self.ui.input_board.text))
        self.ui.input_camera_mode.textChanged.connect(lambda: self.change_event('cameraMode', self.ui.input_camera_mode.text))
        # self.ui.square_size_in.valueChanged.connect(self.change_square_size)
        # self.options['squareSizeCm'] = self.ui.square_size_in.value()
        self.ui.process_but.clicked.connect(self.process_but)
        self.ui.connect_but.clicked.connect(self.camera_connect)

        self.ui.slider_focus.valueChanged.connect(lambda: self.change_event('rgbLensPosition', self.ui.slider_focus.value))
        self.options = {
            'board': self.ui.camera_box.currentText(),
            'squreSizeCm': self.ui.square_size_in.value(),
            'markerSizeCm': self.ui.marker_size_in.value(),
            'fps': self.ui.fps_in.value(),
            'count': 1,
            'defaultBoard': False,
            'squaresX': 11,
            'squaresY': 8,
            'rectifiedDisp': True,
            'disableRgb': False,
            'swapLR': False,
            'mode': ['capture', 'process'],
            'invert_v': False,
            'invert_h': False,
            'maxEpiploarError': 1.0,
            'cameraMode': 'perspective',
            'rgbLensPosition': 135,
            'debug': False,
            'outputScaleFactor': 0.5,
            'captureDelay': 0,
            'factoryCalibration': False}
        print(self.options)

    def process_but(self):
            self.options['mode'] = ['process']
            self.camera_connect()

    def change_event(self, parameter, function):
        self.options[parameter] = function()
        print(function())

    def change_square_size(self):
        self.options['squareSizeCm'] = self.ui.square_size_in.value()

    def camera_connect(self):
        self.ui.connect_but.setDisabled(True)
        # self.ui.mainWindow.repaint()
        if not hasattr(self, 'camera'):

            if len(dai.Device.getAllAvailableDevices()) == 0:
                print('No available device')
                self.ui.connect_but.setEnabled(True)
                time.sleep(0.5)
                return
            mxid = dai.Device.getAllAvailableDevices()[0].mxid
            self.options['squareSizeCm'] = self.ui.square_size_in.value()
            self.options['invert_h'] = self.ui.mirror_check.isChecked()
            self.key_out, self.key_in = Pipe()
            self.image_in, self.image_out = Pipe()
            self.camera = Process(target=run_main_camera, args=(self.options, mxid, self.image_out, self.key_in,))
            self.camera.start()
            self.timer.start(1000 // self.options['fps'])
            time.sleep(0.5)
            print(f'camera {self.camera}')
        else:
            self.ui.connect_but.setDisabled(True)
            self.key_out.send('q')
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
            # self.ui.image.setPixmap(self.camera_pixmap)
            self.ui.connect_but.setText("Connect")
            self.ui.connect_but.setEnabled(True)
            self.timer.stop()

    def update_image(self):
        self.ui.image.setScaledContents(True)
        result = self.image_in.poll(2 / self.options['fps'])
        if self.camera.is_alive():
            if result:
                if not hasattr(self, 'camera'):
                    self.ui.image.setPixmap(self.camera_pixmap)
                    return
                if not self.ui.connect_but.isEnabled():
                    self.ui.connect_but.setEnabled(True)
                    self.ui.connect_but.setText("Disconnect")
                capture = self.image_in.recv()
                capture = self.convert_cv_qt(capture)
                self.ui.image.setPixmap(capture)
                if self.old_display_width != capture.width() or self.old_display_height != capture.height():
                    self.ui.image.resize(capture.width(), capture.height())
                    self.old_display_width = capture.width()
                    self.old_display_height = capture.height()
        else:
            if not self.ui.connect_but.isEnabled():
                self.ui.connect_but.setEnabled(True)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def capture(self):
        self.key_out.send(' ')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    class_instance = Application()
    class_instance.show()
    # MainWindow = QtWidgets.QMainWindow()
    # ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    sys.exit(app.exec_())

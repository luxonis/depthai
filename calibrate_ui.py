from PyQt5 import QtCore, QtGui, QtWidgets

import calibrate2 as calibrate
import cv2
from multiprocessing import Process, Pipe
import time
import depthai


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1072, 908)
        font = QtGui.QFont()
        font.setPointSize(11)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(410, 10, 640, 481))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/oak-d.jpg"))
        self.image.setObjectName("image")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(10, 10, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.logs_box = QtWidgets.QGroupBox(self.centralwidget)
        self.logs_box.setGeometry(QtCore.QRect(20, 640, 1041, 191))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.logs_box.setFont(font)
        self.logs_box.setObjectName("logs_box")
        self.logs = QtWidgets.QTextBrowser(self.logs_box)
        self.logs.setGeometry(QtCore.QRect(10, 30, 1021, 141))
        self.logs.setObjectName("logs")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(20, 50, 341, 571))
        self.tabWidget.setObjectName("tabWidget")
        self.basic_tab = QtWidgets.QWidget()
        self.basic_tab.setObjectName("basic_tab")
        self.select_box = QtWidgets.QGroupBox(self.basic_tab)
        self.select_box.setGeometry(QtCore.QRect(0, 0, 321, 81))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.select_box.setFont(font)
        self.select_box.setObjectName("select_box")
        self.camera_box = QtWidgets.QComboBox(self.select_box)
        self.camera_box.setGeometry(QtCore.QRect(10, 30, 285, 37))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera_box.sizePolicy().hasHeightForWidth())
        self.camera_box.setSizePolicy(sizePolicy)
        self.camera_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.camera_box.setObjectName("camera_box")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/oak-1.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon1, "")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/oak-d-lite.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon2, "")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/oak-d.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon3, "")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/oak-d-pro.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon4, "")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/rpi-compute-mod.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon5, "")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/rpi-hat.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon6, "")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("./../../.designer/backup/Assets/usb3.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera_box.addItem(icon7, "")
        self.l_fps_2 = QtWidgets.QLabel(self.basic_tab)
        self.l_fps_2.setGeometry(QtCore.QRect(130, 160, 31, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.l_fps_2.setFont(font)
        self.l_fps_2.setObjectName("l_fps_2")
        self.rect_check = QtWidgets.QCheckBox(self.basic_tab)
        self.rect_check.setGeometry(QtCore.QRect(20, 320, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.rect_check.setFont(font)
        self.rect_check.setObjectName("rect_check")
        self.fps_in = QtWidgets.QSpinBox(self.basic_tab)
        self.fps_in.setGeometry(QtCore.QRect(170, 150, 51, 31))
        self.fps_in.setMaximum(120)
        self.fps_in.setProperty("value", 30)
        self.fps_in.setObjectName("fps_in")
        self.mirror_check = QtWidgets.QCheckBox(self.basic_tab)
        self.mirror_check.setGeometry(QtCore.QRect(20, 260, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.mirror_check.setFont(font)
        self.mirror_check.setChecked(True)
        self.mirror_check.setObjectName("mirror_check")
        self.l_rgb_lens_pos_2 = QtWidgets.QLabel(self.basic_tab)
        self.l_rgb_lens_pos_2.setGeometry(QtCore.QRect(20, 190, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.l_rgb_lens_pos_2.setFont(font)
        self.l_rgb_lens_pos_2.setObjectName("l_rgb_lens_pos_2")
        self.square_size_in = QtWidgets.QDoubleSpinBox(self.basic_tab)
        self.square_size_in.setGeometry(QtCore.QRect(170, 110, 71, 31))
        self.square_size_in.setSingleStep(0.1)
        self.square_size_in.setProperty("value", 2.0)
        self.square_size_in.setObjectName("square_size_in")
        self.swap_check = QtWidgets.QCheckBox(self.basic_tab)
        self.swap_check.setGeometry(QtCore.QRect(20, 290, 211, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.swap_check.setFont(font)
        self.swap_check.setObjectName("swap_check")
        self.lens_pos_in = QtWidgets.QSpinBox(self.basic_tab)
        self.lens_pos_in.setGeometry(QtCore.QRect(170, 190, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lens_pos_in.setFont(font)
        self.lens_pos_in.setMaximum(300)
        self.lens_pos_in.setProperty("value", 135)
        self.lens_pos_in.setObjectName("lens_pos_in")
        self.l_sqr_size = QtWidgets.QLabel(self.basic_tab)
        self.l_sqr_size.setGeometry(QtCore.QRect(30, 110, 131, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.l_sqr_size.setFont(font)
        self.l_sqr_size.setObjectName("l_sqr_size")
        self.check_show_board = QtWidgets.QCheckBox(self.basic_tab)
        self.check_show_board.setGeometry(QtCore.QRect(20, 230, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.check_show_board.setFont(font)
        self.check_show_board.setObjectName("check_show_board")
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
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.advanced_tab)
        self.doubleSpinBox_3.setGeometry(QtCore.QRect(10, 400, 62, 24))
        self.doubleSpinBox_3.setSingleStep(0.1)
        self.doubleSpinBox_3.setProperty("value", 1.0)
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.l_max_epi = QtWidgets.QLabel(self.advanced_tab)
        self.l_max_epi.setGeometry(QtCore.QRect(80, 400, 191, 16))
        self.l_max_epi.setObjectName("l_max_epi")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.advanced_tab)
        self.lineEdit_3.setGeometry(QtCore.QRect(10, 430, 101, 23))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.l_cam_mode = QtWidgets.QLabel(self.advanced_tab)
        self.l_cam_mode.setGeometry(QtCore.QRect(120, 430, 111, 16))
        self.l_cam_mode.setObjectName("l_cam_mode")
        self.spin_rgb_lens_pos = QtWidgets.QSpinBox(self.advanced_tab)
        self.spin_rgb_lens_pos.setGeometry(QtCore.QRect(10, 460, 51, 24))
        self.spin_rgb_lens_pos.setMaximum(300)
        self.spin_rgb_lens_pos.setProperty("value", 135)
        self.spin_rgb_lens_pos.setObjectName("spin_rgb_lens_pos")
        self.l_rgb_lens_pos = QtWidgets.QLabel(self.advanced_tab)
        self.l_rgb_lens_pos.setGeometry(QtCore.QRect(70, 460, 141, 16))
        self.l_rgb_lens_pos.setObjectName("l_rgb_lens_pos")
        self.spin_fps = QtWidgets.QSpinBox(self.advanced_tab)
        self.spin_fps.setGeometry(QtCore.QRect(10, 490, 43, 24))
        self.spin_fps.setMaximum(120)
        self.spin_fps.setProperty("value", 30)
        self.spin_fps.setObjectName("spin_fps")
        self.l_fps = QtWidgets.QLabel(self.advanced_tab)
        self.l_fps.setGeometry(QtCore.QRect(60, 490, 41, 16))
        self.l_fps.setObjectName("l_fps")
        self.pushButton = QtWidgets.QPushButton(self.advanced_tab)
        self.pushButton.setGeometry(QtCore.QRect(80, 10, 80, 23))
        self.pushButton.setObjectName("pushButton")
        self.l_markersize = QtWidgets.QLabel(self.advanced_tab)
        self.l_markersize.setGeometry(QtCore.QRect(80, 100, 121, 16))
        self.l_markersize.setObjectName("l_markersize")
        self.tabWidget.addTab(self.advanced_tab, "")
        self.calibrate_but = QtWidgets.QPushButton(self.centralwidget)
        self.calibrate_but.setGeometry(QtCore.QRect(540, 520, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.calibrate_but.setFont(font)
        self.calibrate_but.setObjectName("calibrate_but")
        self.connect_but = QtWidgets.QPushButton(self.centralwidget)
        self.connect_but.setGeometry(QtCore.QRect(440, 520, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.connect_but.setFont(font)
        self.connect_but.setObjectName("connect_but")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1072, 34))
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
        self.l_fps_2.setText(_translate("MainWindow", "FPS"))
        self.rect_check.setText(_translate("MainWindow", "Show Rectified images"))
        self.mirror_check.setText(_translate("MainWindow", "Mirror image"))
        self.l_rgb_lens_pos_2.setText(_translate("MainWindow", "RGB Lens Position"))
        self.swap_check.setText(_translate("MainWindow", "Swap Left Right Cameras"))
        self.l_sqr_size.setText(_translate("MainWindow", "Square Size(cm)"))
        self.check_show_board.setText(_translate("MainWindow", "Show charuco board"))
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
        self.input_mode.setText(_translate("MainWindow", "[\'capture\', \'process\']"))
        self.l_mode.setText(_translate("MainWindow", "Mode"))
        self.input_board.setText(_translate("MainWindow", "BW10980BC"))
        self.l_board.setText(_translate("MainWindow", "Board"))
        self.check_inv_vert.setText(_translate("MainWindow", "Invert Vertical"))
        self.check_invert_horz.setText(_translate("MainWindow", "Invert Horizontal"))
        self.l_max_epi.setText(_translate("MainWindow", "maximum Epipolar Error"))
        self.lineEdit_3.setText(_translate("MainWindow", "perspective"))
        self.l_cam_mode.setText(_translate("MainWindow", "Camera Mode"))
        self.l_rgb_lens_pos.setText(_translate("MainWindow", "RGB Lens Position"))
        self.l_fps.setText(_translate("MainWindow", "FPS"))
        self.pushButton.setText(_translate("MainWindow", "Reset"))
        self.l_markersize.setText(_translate("MainWindow", "Markersize Cm"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.advanced_tab), _translate("MainWindow", "Advanced"))
        self.calibrate_but.setText(_translate("MainWindow", "Calibrate"))
        self.connect_but.setText(_translate("MainWindow", "Connect"))

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


def run_main_camera(options, show_img=cv2.imshow, wait_key=cv2.waitKey):
    main = calibrate.Main(options, show_img, wait_key)
    main.run()


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
            'fps': 30,
            'debug': False,
            'outputScaleFactor': 0.5,
            'captureDelay': 2}
        self.args = self.options
        self.display_width = 640
        self.display_height = 480
        self.old_display_width = 640
        self.old_display_height = 480
        self.key = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_image)
        # self.device_test = depthai.Device()
        self.ui.connect_but.clicked.connect(self.camera_connect)
        self.ui.calibrate_but.clicked.connect(self.capture)

    def camera_connect(self):
        self.ui.connect_but.setDisabled(True)
        # self.ui.mainWindow.repaint()
        if not hasattr(self, 'camera'):
            self.options['squareSizeCm'] = self.ui.square_size_in.value()
            self.options['invert_h'] = self.ui.mirror_check.isChecked()
            self.key_out, self.key_in = Pipe()
            self.image_in, self.image_out = Pipe()
            self.camera = Process(target=run_main_camera, args=(self.options, self.image_out, self.key_in,))
            self.camera.start()
            self.timer.start(1000 // self.options['fps'])
        else:
            self.ui.connect_but.setDisabled(True)
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
            self.ui.image.setPixmap(self.camera_pixmap)
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

import sys, threading
from PyQt5 import QtCore, QtGui, QtWidgets
import time

class SplashScreen(QtWidgets.QSplashScreen):
    sigEnableHeartbeat = QtCore.pyqtSignal(bool)
    def __init__(self, filename=None):
        splashSize = QtCore.QSize(400, 400)
        geometry = QtWidgets.QApplication.instance().primaryScreen().geometry()
        if geometry != None:
            splashSize = QtCore.QSize(int(geometry.width() / 4), int(geometry.height() / 4))
        splashImage = QtGui.QPixmap(filename).scaled(splashSize, QtCore.Qt.KeepAspectRatio)
        #self.splash = QtWidgets.QSplashScreen(splashImage, QtCore.Qt.WindowStaysOnTopHint)
        QtWidgets.QSplashScreen.__init__(self, splashImage, QtCore.Qt.WindowStaysOnTopHint)

        # Disable closing on mouse press
        self.mousePressEvent = lambda e : None

        # Signals
        #self.sigEnableHeartbeat.connect(self.internalEnableHeartbeat)

        #self.updateSplashMessage('')
        self.show()
        self.running = True
        self.heartbeatAnimation = False
        self.animationThread = threading.Thread(target=self.animation)
        self.animationThread.start()

    def __del__(self):
        self.running = False
    #    self.animationThread.join()

    def animation(self):
        heartbeatCycle = 1.0
        heartbeatCycleDelta = -0.02
        while self.running is True:
            if self.heartbeatAnimation:
                heartbeatCycle = heartbeatCycle + heartbeatCycleDelta
                if heartbeatCycle <= 0.7 or heartbeatCycle >= 1.0:
                    heartbeatCycleDelta = -heartbeatCycleDelta
                self.setOpacity(heartbeatCycle)

            # Process events and sleep
            #self.processEvents()
            time.sleep(0.1) # 10 FPS

    # @QtCore.pyqtSlot(bool)
    # def internalEnableHeartbeat(self, enable):
    #     #self.setOpacity(1.0)
    #     self.heartbeatAnimation = enable

    def enableHeartbeat(self, enable):
        self.setOpacity(1.0)
        self.heartbeatAnimation = enable
        #self.sigEnableHeartbeat.emit(enable)

    @QtCore.pyqtSlot(float)
    def setOpacity(self, opacity):
        self.setWindowOpacity(opacity)
        self.repaint()

    @QtCore.pyqtSlot(str)
    def updateSplashMessage(self, msg=''):
        self.showMessage("%s" % msg.title(), QtCore.Qt.AlignBottom)

    @QtCore.pyqtSlot()
    def show(self):
        super().show()

    @QtCore.pyqtSlot()
    def hide(self):
        self.enableHeartbeat(False)
        super().hide()

    @QtCore.pyqtSlot()
    def close(self):
        super().close()
        self.running = False

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(['DepthAI Launcher'])
    splashImage = 'splash2.png'
    splashScreen = SplashScreen(splashImage)
    sys.exit(qApp.exec_())

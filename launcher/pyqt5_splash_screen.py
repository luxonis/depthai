import sys, threading
from PyQt5 import QtCore, QtGui, QtWidgets
import time

class SplashScreen():
    def __init__(self, filename=None):
        splashSize = QtCore.QSize(400, 400)
        geometry = QtWidgets.QApplication.instance().primaryScreen().geometry()
        if geometry != None:
            splashSize = QtCore.QSize(int(geometry.width() / 4), int(geometry.height() / 4))
        splashImage = QtGui.QPixmap(filename).scaled(splashSize, QtCore.Qt.KeepAspectRatio)
        self.splash = QtWidgets.QSplashScreen(splashImage, QtCore.Qt.WindowStaysOnTopHint)
        self.splash.mousePressEvent = lambda e : None
        #self.updateSplashMessage('')
        self.splash.show()
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

    def enableHeartbeat(self, enable):
        self.setOpacity(1.0)
        self.heartbeatAnimation = enable

    @QtCore.pyqtSlot(float)
    def setOpacity(self, opacity):
        self.splash.setWindowOpacity(opacity)
        self.splash.repaint()

    @QtCore.pyqtSlot(str)
    def updateSplashMessage(self, msg=''):
        self.splash.showMessage("%s" % msg.title(), QtCore.Qt.AlignBottom)

    @QtCore.pyqtSlot()
    def show(self):
        self.splash.show()

    @QtCore.pyqtSlot()
    def hide(self):
        self.enableHeartbeat(False)
        self.splash.hide()

    @QtCore.pyqtSlot()
    def close(self):
        self.splash.close()
        self.running = False

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(['DepthAI Launcher'])
    splashImage = 'splash2.png'
    splashScreen = SplashScreen(splashImage)
    sys.exit(qApp.exec_())

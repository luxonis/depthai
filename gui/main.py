# This Python file uses the following encoding: utf-8
import sys
import threading
from pathlib import Path
import time
from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
import depthai as dai

# To be used on the @QmlElement decorator
# (QML_IMPORT_MINOR_VERSION is optional)
QML_IMPORT_NAME = "dai.gui"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class DepthBridge(QObject):
    onUpdate = Signal()

    @Slot(bool)
    def toggleSubpixel(self, state):
        print("Sub: {}".format(state))

    @Slot(bool)
    def toggleExtendedDisparity(self, state):
        print("Ext: {}".format(state))

    @Slot(bool)
    def toggleLeftRightCheck(self, state):
        print("Lrc: {}".format(state))

    @Slot(int)
    def setDisparityConfidenceThreshold(self, value):
        print("Dct: {}".format(value))

    @Slot(int)
    def setBilateralSigma(self, value):
        print("Bls: {}".format(value))

    @Slot(int)
    def setBilateralSigma(self, value):
        print("Sig: {}".format(value))

    @Slot(int, int)
    def setDepthRange(self, valFrom, valTo):
        print("Rng: {} - {}".format(valFrom, valTo))

    @Slot(str)
    def setMedianFilter(self, state):
        print("Med: {}".format(state))


class BaseCamBridge(QObject):
    @Slot(int)
    def setIso(self, value):
        print("ISO: {}".format(value))

    @Slot(int)
    def setExposure(self, value):
        print("Exposure: {}".format(value))

    @Slot(int)
    def setContrast(self, value):
        print("Contrast: {}".format(value))

    @Slot(int)
    def setBrightness(self, value):
        print("Brightness: {}".format(value))

    @Slot(int)
    def setSaturation(self, value):
        print("Saturation: {}".format(value))

    @Slot(int)
    def setSharpness(self, value):
        print("Sharpness: {}".format(value))


@QmlElement
class ColorCamBridge(BaseCamBridge):
    onUpdate = Signal()


@QmlElement
class LeftCamBridge(BaseCamBridge):
    onUpdate = Signal()


@QmlElement
class RightCamBridge(BaseCamBridge):
    onUpdate = Signal()


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class DemoQtGui:
    def __init__(self):
        self.app = QGuiApplication()
        self.engine = QQmlApplicationEngine()

    def setData(self, name, value):
        self.engine.rootContext().setContextProperty(name, value)

    def startGui(self):
        self.engine.load(Path(__file__).parent / "views" / "root.qml")
        if not self.engine.rootObjects():
            raise RuntimeError("Unable to start GUI - no root objects!")
        sys.exit(self.app.exec())


if __name__ == "__main__":
    medianChoices = list(filter(lambda name: name.startswith('KERNEL_') or name.startswith('MEDIAN_'), vars(dai.MedianFilter).keys()))[::-1]
    gui = DemoQtGui()
    gui.setData("medianChoices", medianChoices)
    gui.startGui()

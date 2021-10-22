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


class DemoQtGui:
    instance = None

    def __init__(self):
        self.app = QGuiApplication()
        self.engine = QQmlApplicationEngine()
        self.setInstance()

    def setInstance(self):
        DemoQtGui.instance = self

    def setData(self, name, value):
        self.engine.rootContext().setContextProperty(name, value)

    def startGui(self):
        self.engine.load(Path(__file__).parent / "views" / "root.qml")
        if not self.engine.rootObjects():
            raise RuntimeError("Unable to start GUI - no root objects!")
        sys.exit(self.app.exec())

    def guiOnDepthConfigUpdate(self, median=None):
        pass


@QmlElement
class DepthBridge(QObject):
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
        value = getattr(dai.MedianFilter, state)
        DemoQtGui.instance.guiOnDepthConfigUpdate(median=value)
        print("Med: {}".format(value))


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
    pass


@QmlElement
class LeftCamBridge(BaseCamBridge):
    pass


@QmlElement
class RightCamBridge(BaseCamBridge):
    pass


if __name__ == "__main__":
    medianChoices = list(filter(lambda name: name.startswith('KERNEL_') or name.startswith('MEDIAN_'), vars(dai.MedianFilter).keys()))[::-1]
    gui = DemoQtGui()
    gui.setData("medianChoices", medianChoices)
    gui.startGui()

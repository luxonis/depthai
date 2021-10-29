# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path
from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
import depthai as dai

# To be used on the @QmlElement decorator
# (QML_IMPORT_MINOR_VERSION is optional)
from PySide6.QtQuick import QQuickPaintedItem
from PySide6.QtWidgets import QMessageBox, QApplication

QML_IMPORT_NAME = "dai.gui"
QML_IMPORT_MAJOR_VERSION = 1

class Singleton(type(QQuickPaintedItem)):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@QmlElement
class ImageWriter(QQuickPaintedItem, metaclass=Singleton):
    frame = QImage()

    def __init__(self):
        super().__init__()
        self.setRenderTarget(QQuickPaintedItem.FramebufferObject)

    def paint(self, painter):
        painter.drawImage(0, 0, self.frame)

    def update_frame(self, image):
        self.frame = image
        self.update()


class DemoQtGui:
    instance = None
    writer = None
    window = None

    def __init__(self):
        self.app = QApplication()
        self.engine = QQmlApplicationEngine()
        self.setInstance()
        self.engine.load(Path(__file__).parent / "views" / "root.qml")
        self.window = self.engine.rootObjects()[0]
        if not self.engine.rootObjects():
            raise RuntimeError("Unable to start GUI - no root objects!")

    def setInstance(self):
        DemoQtGui.instance = self

    @Slot(list)
    def setData(self, data):
        name, value = data
        self.window.setProperty(name, value)

    @Slot(QImage)
    def updatePreview(self, data):
        self.writer.update_frame(data)

    def startGui(self):
        self.writer = ImageWriter()
        return self.app.exec()


@QmlElement
class AppBridge(QObject):
    @Slot()
    def applyAndRestart(self):
        DemoQtGui.instance.restartDemo()


@QmlElement
class AIBridge(QObject):
    @Slot(str)
    def setCnnModel(self, name):
        DemoQtGui.instance.guiOnAiSetupUpdate(cnn=name)

    @Slot(int)
    def setShaves(self, value):
        DemoQtGui.instance.guiOnAiSetupUpdate(shave=value)

    @Slot(str)
    def setModelSource(self, value):
        DemoQtGui.instance.guiOnAiSetupUpdate(source=value)

    @Slot(bool)
    def setFullFov(self, value):
        DemoQtGui.instance.guiOnAiSetupUpdate(fullFov=value)

    @Slot(bool)
    def setSyncNN(self, value):
        DemoQtGui.instance.guiOnAiSetupUpdate(sync=value)

    @Slot(bool)
    def setSbb(self, value):
        DemoQtGui.instance.guiOnAiSetupUpdate(sbb=value)

    @Slot(float)
    def setSbbFactor(self, value):
        if DemoQtGui.writer is not None:
            DemoQtGui.instance.guiOnAiSetupUpdate(sbbFactor=value)

    @Slot(str)
    def setOvVersion(self, state):
        DemoQtGui.instance.guiOnAiSetupUpdate(ov=state.replace("VERSION_", ""))

    @Slot(str)
    def setCountLabel(self, state):
        DemoQtGui.instance.guiOnAiSetupUpdate(countLabel=state)


@QmlElement
class PreviewBridge(QObject):
    @Slot(str)
    def changeSelected(self, state):
        DemoQtGui.instance.guiOnPreviewChangeSelected(state)


@QmlElement
class DepthBridge(QObject):
    @Slot(bool)
    def toggleSubpixel(self, state):
        DemoQtGui.instance.guiOnDepthSetupUpdate(subpixel=state)

    @Slot(bool)
    def toggleExtendedDisparity(self, state):
        DemoQtGui.instance.guiOnDepthSetupUpdate(extended=state)

    @Slot(bool)
    def toggleLeftRightCheck(self, state):
        DemoQtGui.instance.guiOnDepthSetupUpdate(lrc=state)

    @Slot(int)
    def setDisparityConfidenceThreshold(self, value):
        DemoQtGui.instance.guiOnDepthConfigUpdate(dct=value)

    @Slot(int)
    def setLrcThreshold(self, value):
        DemoQtGui.instance.guiOnDepthConfigUpdate(lrcThreshold=value)

    @Slot(int)
    def setBilateralSigma(self, value):
        DemoQtGui.instance.guiOnDepthConfigUpdate(sigma=value)

    @Slot(int, int)
    def setDepthRange(self, valFrom, valTo):
        DemoQtGui.instance.guiOnDepthSetupUpdate(depthFrom=valFrom, depthTo=valTo)

    @Slot(str)
    def setMedianFilter(self, state):
        value = getattr(dai.MedianFilter, state)
        DemoQtGui.instance.guiOnDepthConfigUpdate(median=value)


class BaseCamBridge(QObject):
    name = "base"

    @Slot(int, int)
    def setIsoExposure(self, iso, exposure):
        if iso > 0 and exposure > 0:
            DemoQtGui.instance.guiOnCameraConfigUpdate(self.name, sensitivity=iso, exposure=exposure)

    @Slot(int)
    def setContrast(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate(self.name, contrast=value)

    @Slot(int)
    def setBrightness(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate(self.name, brightness=value)

    @Slot(int)
    def setSaturation(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate(self.name, saturation=value)

    @Slot(int)
    def setSharpness(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate(self.name, sharpness=value)

    @Slot(int)
    def setFps(self, value):
        DemoQtGui.instance.guiOnCameraSetupUpdate(self.name, fps=value)

    @Slot(str)
    def setResolution(self, state):
        if state == "THE_1080_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate(self.name, resolution=1080)
        elif state == "THE_4_K":
            DemoQtGui.instance.guiOnCameraSetupUpdate(self.name, resolution=2160)
        elif state == "THE_12_MP":
            DemoQtGui.instance.guiOnCameraSetupUpdate(self.name, resolution=3040)
        elif state == "THE_720_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate(self.name, resolution=720)
        elif state == "THE_800_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate(self.name, resolution=800)
        elif state == "THE_400_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate(self.name, resolution=400)


@QmlElement
class ColorCamBridge(BaseCamBridge):
    name = "color"


@QmlElement
class LeftCamBridge(BaseCamBridge):
    name = "left"


@QmlElement
class RightCamBridge(BaseCamBridge):
    name = "right"


if __name__ == "__main__":
    medianChoices = list(filter(lambda name: name.startswith('KERNEL_') or name.startswith('MEDIAN_'), vars(dai.MedianFilter).keys()))[::-1]
    gui = DemoQtGui()
    gui.setData("medianChoices", medianChoices)
    gui.startGui()

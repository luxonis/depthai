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

    def __init__(self):
        self.app = QGuiApplication()
        self.engine = QQmlApplicationEngine()
        self.setInstance()

    def setInstance(self):
        DemoQtGui.instance = self

    @Slot(list)
    def setData(self, data):
        name, value = data
        self.engine.rootContext().setContextProperty(name, value)

    @Slot(QImage)
    def updatePreview(self, data):
        self.writer.update_frame(data)

    def startGui(self):
        self.engine.load(Path(__file__).parent / "views" / "root.qml")
        if not self.engine.rootObjects():
            raise RuntimeError("Unable to start GUI - no root objects!")
        self.writer = ImageWriter()
        return self.app.exec()


@QmlElement
class AppBridge(QObject):
    @Slot()
    def applyAndRestart(self):
        DemoQtGui.instance.restartDemo()


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

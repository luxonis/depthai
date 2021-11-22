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
from depthai_sdk import Previews

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
        medianChoices = list(filter(lambda name: name.startswith('KERNEL_') or name.startswith('MEDIAN_'), vars(dai.MedianFilter).keys()))[::-1]
        self.setData(["medianChoices", medianChoices])
        colorChoices = list(filter(lambda name: name[0].isupper(), vars(dai.ColorCameraProperties.SensorResolution).keys()))
        self.setData(["colorResolutionChoices", colorChoices])
        monoChoices = list(filter(lambda name: name[0].isupper(), vars(dai.MonoCameraProperties.SensorResolution).keys()))
        self.setData(["monoResolutionChoices", monoChoices])
        self.setData(["modelSourceChoices", [Previews.color.name, Previews.left.name, Previews.right.name]])
        versionChoices = sorted(filter(lambda name: name.startswith("VERSION_"), vars(dai.OpenVINO).keys()), reverse=True)
        self.setData(["ovVersions", versionChoices])
        return self.app.exec()


@QmlElement
class AppBridge(QObject):
    @Slot()
    def applyAndRestart(self):
        DemoQtGui.instance.restartDemo()

    @Slot()
    def reloadDevices(self):
        DemoQtGui.instance.guiOnReloadDevices()

    @Slot(bool)
    def toggleStatisticsConsent(self, value):
        DemoQtGui.instance.guiOnStaticticsConsent(value)

    @Slot(str)
    def selectDevice(self, value):
        DemoQtGui.instance.guiOnSelectDevice(value)

    @Slot(bool, bool, bool)
    def selectReportingOptions(self, temp, cpu, memory):
        DemoQtGui.instance.guiOnSelectReportingOptions(temp, cpu, memory)

    @Slot(str)
    def selectReportingPath(self, value):
        DemoQtGui.instance.guiOnSelectReportingPath(value)

    @Slot(str)
    def selectEncodingPath(self, value):
        DemoQtGui.instance.guiOnSelectEncodingPath(value)

    @Slot(bool, int)
    def toggleColorEncoding(self, enabled, fps):
        DemoQtGui.instance.guiOnToggleColorEncoding(enabled, fps)

    @Slot(bool, int)
    def toggleLeftEncoding(self, enabled, fps):
        DemoQtGui.instance.guiOnToggleLeftEncoding(enabled, fps)

    @Slot(bool, int)
    def toggleRightEncoding(self, enabled, fps):
        DemoQtGui.instance.guiOnToggleRightEncoding(enabled, fps)

    @Slot(bool)
    def toggleDepth(self, enabled):
        DemoQtGui.instance.guiOnToggleDepth(enabled)

    @Slot(bool)
    def toggleNN(self, enabled):
        DemoQtGui.instance.guiOnToggleNN(enabled)

    @Slot(bool)
    def toggleDisparity(self, enabled):
        DemoQtGui.instance.guiOnToggleDisparity(enabled)


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
    def setSbb(self, value):
        DemoQtGui.instance.guiOnAiSetupUpdate(sbb=value)

    @Slot(float)
    def setSbbFactor(self, value):
        if DemoQtGui.instance.writer is not None:
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
        DemoQtGui.instance.guiOnDepthConfigUpdate(lrc=state)

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


@QmlElement
class ColorCamBridge(QObject):
    name = "color"

    @Slot(int, int)
    def setIsoExposure(self, iso, exposure):
        if iso > 0 and exposure > 0:
            DemoQtGui.instance.guiOnCameraConfigUpdate("color", sensitivity=iso, exposure=exposure)

    @Slot(int)
    def setContrast(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("color", contrast=value)

    @Slot(int)
    def setBrightness(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("color", brightness=value)

    @Slot(int)
    def setSaturation(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("color", saturation=value)

    @Slot(int)
    def setSharpness(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("color", sharpness=value)

    @Slot(int)
    def setFps(self, value):
        DemoQtGui.instance.guiOnCameraSetupUpdate("color", fps=value)

    @Slot(str)
    def setResolution(self, state):
        if state == "THE_1080_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate("color", resolution=1080)
        elif state == "THE_4_K":
            DemoQtGui.instance.guiOnCameraSetupUpdate("color", resolution=2160)
        elif state == "THE_12_MP":
            DemoQtGui.instance.guiOnCameraSetupUpdate("color", resolution=3040)


@QmlElement
class MonoCamBridge(QObject):

    @Slot(int, int)
    def setIsoExposure(self, iso, exposure):
        if iso > 0 and exposure > 0:
            DemoQtGui.instance.guiOnCameraConfigUpdate("left", sensitivity=iso, exposure=exposure)
            DemoQtGui.instance.guiOnCameraConfigUpdate("right", sensitivity=iso, exposure=exposure)

    @Slot(int)
    def setContrast(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("left", contrast=value)
        DemoQtGui.instance.guiOnCameraConfigUpdate("right", contrast=value)

    @Slot(int)
    def setBrightness(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("left", brightness=value)
        DemoQtGui.instance.guiOnCameraConfigUpdate("right", brightness=value)

    @Slot(int)
    def setSaturation(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("left", saturation=value)
        DemoQtGui.instance.guiOnCameraConfigUpdate("right", saturation=value)

    @Slot(int)
    def setSharpness(self, value):
        DemoQtGui.instance.guiOnCameraConfigUpdate("left", sharpness=value)
        DemoQtGui.instance.guiOnCameraConfigUpdate("right", sharpness=value)

    @Slot(int)
    def setFps(self, value):
        DemoQtGui.instance.guiOnCameraSetupUpdate("left", fps=value)
        DemoQtGui.instance.guiOnCameraSetupUpdate("right", fps=value)

    @Slot(str)
    def setResolution(self, state):
        if state == "THE_720_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate("left", resolution=720)
            DemoQtGui.instance.guiOnCameraSetupUpdate("right", resolution=720)
        elif state == "THE_800_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate("left", resolution=800)
            DemoQtGui.instance.guiOnCameraSetupUpdate("right", resolution=800)
        elif state == "THE_400_P":
            DemoQtGui.instance.guiOnCameraSetupUpdate("left", resolution=400)
            DemoQtGui.instance.guiOnCameraSetupUpdate("right", resolution=400)

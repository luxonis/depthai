# This Python file uses the following encoding: utf-8
import argparse
import json
import sys
import time
import traceback
from functools import cmp_to_key
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtQml import QQmlApplicationEngine, qmlRegisterType
from PyQt5.QtQuick import QQuickPaintedItem
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
import depthai as dai

# To be used on the @QmlElement decorator
# (QML_IMPORT_MINOR_VERSION is optional)
from PyQt5.QtWidgets import QApplication
from depthai_sdk import Previews, resizeLetterbox, createBlankFrame, loadModule

from depthai_helpers.config_manager import prepareConfManager


class Singleton(type(QQuickPaintedItem)):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


instance = None


# @QmlElement
class ImageWriter(QQuickPaintedItem):
    frame = QImage()

    def __init__(self, parent):
        super().__init__(parent)
        self.setRenderTarget(QQuickPaintedItem.FramebufferObject)
        self.setProperty("parent", parent)

    def paint(self, painter):
        painter.drawImage(0, 0, self.frame)

    def update_frame(self, image):
        self.frame = image
        self.update()


# @QmlElement
class AppBridge(QObject):
    @pyqtSlot()
    def applyAndRestart(self):
        instance.restartDemo()

    @pyqtSlot()
    def reloadDevices(self):
        instance.guiOnReloadDevices()

    @pyqtSlot(bool)
    def toggleStatisticsConsent(self, value):
        instance.guiOnStaticticsConsent(value)

    @pyqtSlot(bool)
    def guiOnToggleSync(self, value):
        self.updateArg("sync", value)

    @pyqtSlot(str)
    def runApp(self, appName):
        instance.guiOnRunApp(appName)

    @pyqtSlot(str)
    def terminateApp(self, appName):
        instance.guiOnTerminateApp(appName)

    @pyqtSlot(str)
    def selectDevice(self, value):
        instance.guiOnSelectDevice(value)

    @pyqtSlot(bool, bool, bool)
    def selectReportingOptions(self, temp, cpu, memory):
        instance.guiOnSelectReportingOptions(temp, cpu, memory)

    @pyqtSlot(str)
    def selectReportingPath(self, value):
        instance.guiOnSelectReportingPath(value)

    @pyqtSlot(str)
    def selectEncodingPath(self, value):
        instance.guiOnSelectEncodingPath(value)

    @pyqtSlot(bool, int)
    def toggleColorEncoding(self, enabled, fps):
        instance.guiOnToggleColorEncoding(enabled, fps)

    @pyqtSlot(bool, int)
    def toggleLeftEncoding(self, enabled, fps):
        instance.guiOnToggleLeftEncoding(enabled, fps)

    @pyqtSlot(bool, int)
    def toggleRightEncoding(self, enabled, fps):
        instance.guiOnToggleRightEncoding(enabled, fps)

    @pyqtSlot(bool)
    def toggleDepth(self, enabled):
        instance.guiOnToggleDepth(enabled)

    @pyqtSlot(bool)
    def toggleNN(self, enabled):
        instance.guiOnToggleNN(enabled)

    @pyqtSlot(bool)
    def toggleDisparity(self, enabled):
        instance.guiOnToggleDisparity(enabled)


# @QmlElement
class AIBridge(QObject):
    @pyqtSlot(str)
    def setCnnModel(self, name):
        instance.guiOnAiSetupUpdate(cnn=name)

    @pyqtSlot(int)
    def setShaves(self, value):
        instance.guiOnAiSetupUpdate(shave=value)

    @pyqtSlot(str)
    def setModelSource(self, value):
        instance.guiOnAiSetupUpdate(source=value)

    @pyqtSlot(bool)
    def setFullFov(self, value):
        instance.guiOnAiSetupUpdate(fullFov=value)

    @pyqtSlot(bool)
    def setSbb(self, value):
        instance.guiOnAiSetupUpdate(sbb=value)

    @pyqtSlot(float)
    def setSbbFactor(self, value):
        if instance.writer is not None:
            instance.guiOnAiSetupUpdate(sbbFactor=value)

    @pyqtSlot(str)
    def setOvVersion(self, state):
        instance.guiOnAiSetupUpdate(ov=state.replace("VERSION_", ""))

    @pyqtSlot(str)
    def setCountLabel(self, state):
        instance.guiOnAiSetupUpdate(countLabel=state)


# @QmlElement
class PreviewBridge(QObject):
    @pyqtSlot(str)
    def changeSelected(self, state):
        instance.guiOnPreviewChangeSelected(state)


# @QmlElement
class DepthBridge(QObject):
    @pyqtSlot(bool)
    def toggleSubpixel(self, state):
        instance.guiOnDepthSetupUpdate(subpixel=state)

    @pyqtSlot(bool)
    def toggleExtendedDisparity(self, state):
        instance.guiOnDepthSetupUpdate(extended=state)

    @pyqtSlot(bool)
    def toggleLeftRightCheck(self, state):
        instance.guiOnDepthConfigUpdate(lrc=state)

    @pyqtSlot(int)
    def setDisparityConfidenceThreshold(self, value):
        instance.guiOnDepthConfigUpdate(dct=value)

    @pyqtSlot(int)
    def setLrcThreshold(self, value):
        instance.guiOnDepthConfigUpdate(lrcThreshold=value)

    @pyqtSlot(int)
    def setBilateralSigma(self, value):
        instance.guiOnDepthConfigUpdate(sigma=value)

    @pyqtSlot(int, int)
    def setDepthRange(self, valFrom, valTo):
        instance.guiOnDepthSetupUpdate(depthFrom=int(valFrom * 1000), depthTo=int(valTo * 1000))

    @pyqtSlot(str)
    def setMedianFilter(self, state):
        value = getattr(dai.MedianFilter, state)
        instance.guiOnDepthConfigUpdate(median=value)


# @QmlElement
class ColorCamBridge(QObject):
    name = "color"

    @pyqtSlot(int, int)
    def setIsoExposure(self, iso, exposure):
        if iso > 0 and exposure > 0:
            instance.guiOnCameraConfigUpdate("color", sensitivity=iso, exposure=exposure)

    @pyqtSlot(int)
    def setContrast(self, value):
        instance.guiOnCameraConfigUpdate("color", contrast=value)

    @pyqtSlot(int)
    def setBrightness(self, value):
        instance.guiOnCameraConfigUpdate("color", brightness=value)

    @pyqtSlot(int)
    def setSaturation(self, value):
        instance.guiOnCameraConfigUpdate("color", saturation=value)

    @pyqtSlot(int)
    def setSharpness(self, value):
        instance.guiOnCameraConfigUpdate("color", sharpness=value)

    @pyqtSlot(int)
    def setFps(self, value):
        instance.guiOnCameraSetupUpdate("color", fps=value)

    @pyqtSlot(str)
    def setResolution(self, state):
        if state == "THE_1080_P":
            instance.guiOnCameraSetupUpdate("color", resolution=1080)
        elif state == "THE_4_K":
            instance.guiOnCameraSetupUpdate("color", resolution=2160)
        elif state == "THE_12_MP":
            instance.guiOnCameraSetupUpdate("color", resolution=3040)


# @QmlElement
class MonoCamBridge(QObject):

    @pyqtSlot(int, int)
    def setIsoExposure(self, iso, exposure):
        if iso > 0 and exposure > 0:
            instance.guiOnCameraConfigUpdate("left", sensitivity=iso, exposure=exposure)
            instance.guiOnCameraConfigUpdate("right", sensitivity=iso, exposure=exposure)

    @pyqtSlot(int)
    def setContrast(self, value):
        instance.guiOnCameraConfigUpdate("left", contrast=value)
        instance.guiOnCameraConfigUpdate("right", contrast=value)

    @pyqtSlot(int)
    def setBrightness(self, value):
        instance.guiOnCameraConfigUpdate("left", brightness=value)
        instance.guiOnCameraConfigUpdate("right", brightness=value)

    @pyqtSlot(int)
    def setSaturation(self, value):
        instance.guiOnCameraConfigUpdate("left", saturation=value)
        instance.guiOnCameraConfigUpdate("right", saturation=value)

    @pyqtSlot(int)
    def setSharpness(self, value):
        instance.guiOnCameraConfigUpdate("left", sharpness=value)
        instance.guiOnCameraConfigUpdate("right", sharpness=value)

    @pyqtSlot(int)
    def setFps(self, value):
        instance.guiOnCameraSetupUpdate("left", fps=value)
        instance.guiOnCameraSetupUpdate("right", fps=value)

    @pyqtSlot(str)
    def setResolution(self, state):
        if state == "THE_720_P":
            instance.guiOnCameraSetupUpdate("left", resolution=720)
            instance.guiOnCameraSetupUpdate("right", resolution=720)
        elif state == "THE_800_P":
            instance.guiOnCameraSetupUpdate("left", resolution=800)
            instance.guiOnCameraSetupUpdate("right", resolution=800)
        elif state == "THE_400_P":
            instance.guiOnCameraSetupUpdate("left", resolution=400)
            instance.guiOnCameraSetupUpdate("right", resolution=400)


class DemoQtGui:
    instance = None
    writer = None
    window = None
    progressFrame = None

    def __init__(self):
        global instance
        self.app = QApplication([sys.argv[0]])
        self.engine = QQmlApplicationEngine()
        self.engine.quit.connect(self.app.quit)
        instance = self
        qmlRegisterType(ImageWriter, 'dai.gui', 1, 0, 'ImageWriter')
        qmlRegisterType(AppBridge, 'dai.gui', 1, 0, 'AppBridge')
        qmlRegisterType(AIBridge, 'dai.gui', 1, 0, 'AIBridge')
        qmlRegisterType(PreviewBridge, 'dai.gui', 1, 0, 'PreviewBridge')
        qmlRegisterType(DepthBridge, 'dai.gui', 1, 0, 'DepthBridge')
        qmlRegisterType(ColorCamBridge, 'dai.gui', 1, 0, 'ColorCamBridge')
        qmlRegisterType(MonoCamBridge, 'dai.gui', 1, 0, 'MonoCamBridge')
        self.engine.addImportPath(str(Path(__file__).parent / "views"))
        self.engine.load(str(Path(__file__).parent / "views" / "root.qml"))
        self.window = self.engine.rootObjects()[0]
        if not self.engine.rootObjects():
            raise RuntimeError("Unable to start GUI - no root objects!")

    def setData(self, data):
        name, value = data
        self.window.setProperty(name, value)

    def updatePreview(self, frame):
        w, h = int(self.writer.width()), int(self.writer.height())
        scaledFrame = resizeLetterbox(frame, (w, h))
        if len(frame.shape) == 3:
            img = QImage(scaledFrame.data, w, h, frame.shape[2] * w, 29)  # 29 - QImage.Format_BGR888
        else:
            img = QImage(scaledFrame.data, w, h, w, 24)  # 24 - QImage.Format_Grayscale8
        self.writer.update_frame(img)

    def updateDownloadProgress(self, curr, total):
        frame = self.createProgressFrame(curr / total)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1], 29)  # 29 - QImage.Format_BGR888
        self.writer.update_frame(img)

    def createProgressFrame(self, donePercentage=None):
        confManager = getattr(self, "confManager", None)
        w, h = int(self.writer.width()), int(self.writer.height())
        if self.progressFrame is None:
            self.progressFrame = createBlankFrame(w, h)
            downloadText = "Downloading model blob..."
            textsize = cv2.getTextSize(downloadText, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 4)[0][0]
            offset = int((w - textsize) / 2)
            cv2.putText(self.progressFrame, downloadText, (offset, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(self.progressFrame, downloadText, (offset, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        newFrame = self.progressFrame.copy()
        if donePercentage is not None:
            cv2.rectangle(newFrame, (100, 300), (460, 350), (255, 255, 255), cv2.FILLED)
            cv2.rectangle(newFrame, (110, 310), (int(110 + 340 * donePercentage), 340), (0, 0, 0), cv2.FILLED)
        return newFrame

    def showSetupFrame(self, text):
        w, h = int(self.writer.width()), int(self.writer.height())
        setupFrame = createBlankFrame(w, h)
        cv2.putText(setupFrame, text, (200, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(setupFrame, text, (200, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img = QImage(setupFrame.data, w, h, setupFrame.shape[2] * w, 29)  # 29 - QImage.Format_BGR888
        self.writer.update_frame(img)

    def startGui(self):
        self.writer = self.window.findChild(QObject, "writer")
        self.showSetupFrame("Starting demo...")
        medianChoices = list(filter(lambda name: name.startswith('KERNEL_') or name.startswith('MEDIAN_'), vars(dai.MedianFilter).keys()))[::-1]
        self.setData(["medianChoices", medianChoices])
        colorChoices = list(filter(lambda name: name[0].isupper(), vars(dai.ColorCameraProperties.SensorResolution).keys()))
        self.setData(["colorResolutionChoices", colorChoices])
        monoChoices = list(filter(lambda name: name[0].isupper(), vars(dai.MonoCameraProperties.SensorResolution).keys()))
        self.setData(["monoResolutionChoices", monoChoices])
        self.setData(["modelSourceChoices", [Previews.color.name, Previews.left.name, Previews.right.name]])
        versionChoices = sorted(filter(lambda name: name.startswith("VERSION_"), vars(dai.OpenVINO).keys()), reverse=True)
        self.setData(["ovVersions", versionChoices])
        self.createProgressFrame()
        return self.app.exec()

class WorkerSignals(QObject):
    updateConfSignal = pyqtSignal(list)
    updateDownloadProgressSignal = pyqtSignal(int, int)
    updatePreviewSignal = pyqtSignal(np.ndarray)
    setDataSignal = pyqtSignal(list)
    exitSignal = pyqtSignal()
    errorSignal = pyqtSignal(str)

class Worker(QRunnable):
    def __init__(self, instance, parent, conf, selectedPreview=None):
        super(Worker, self).__init__()
        self.running = False
        self.selectedPreview = selectedPreview
        self.instance = instance
        self.parent = parent
        self.conf = conf
        self.callback_module = loadModule(conf.args.callback)
        self.file_callbacks = {
            callbackName: getattr(self.callback_module, callbackName)
            for callbackName in ["shouldRun", "onNewFrame", "onShowFrame", "onNn", "onReport", "onSetup", "onTeardown", "onIter"]
            if callable(getattr(self.callback_module, callbackName, None))
        }
        self.instance.setCallbacks(**self.file_callbacks)
        self.signals = WorkerSignals()
        self.signals.exitSignal.connect(self.terminate)
        self.signals.updateConfSignal.connect(self.updateConf)


    def run(self):
        self.running = True
        self.signals.setDataSignal.emit(["restartRequired", False])
        self.instance.setCallbacks(shouldRun=self.shouldRun, onShowFrame=self.onShowFrame, onSetup=self.onSetup, onAppSetup=self.onAppSetup, onAppStart=self.onAppStart, showDownloadProgress=self.showDownloadProgress)
        self.conf.args.bandwidth = "auto"
        if self.conf.args.deviceId is None:
            devices = dai.Device.getAllAvailableDevices()
            if len(devices) > 0:
                defaultDevice = next(map(
                    lambda info: info.getMxId(),
                    filter(lambda info: info.desc.protocol == dai.XLinkProtocol.X_LINK_USB_VSC, devices)
                ), None)
                if defaultDevice is None:
                    defaultDevice = devices[0].getMxId()
                self.conf.args.deviceId = defaultDevice
        self.conf.args.show = [
            Previews.color.name, Previews.nnInput.name, Previews.depth.name, Previews.depthRaw.name, Previews.left.name,
            Previews.rectifiedLeft.name, Previews.right.name, Previews.rectifiedRight.name
        ]
        try:
            self.instance.run_all(self.conf)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as ex:
            self.onError(ex)

    def terminate(self):
        self.running = False
        self.signals.setDataSignal.emit(["restartRequired", False])


    def updateConf(self, argsList):
        self.conf.args = argparse.Namespace(**dict(argsList))

    def onError(self, ex: Exception):
        self.signals.errorSignal.emit(''.join(traceback.format_tb(ex.__traceback__) + [str(ex)]))
        self.signals.setDataSignal.emit(["restartRequired", True])

    def shouldRun(self):
        if "shouldRun" in self.file_callbacks:
            return self.running and self.file_callbacks["shouldRun"]()
        return self.running

    def onShowFrame(self, frame, source):
        if "onShowFrame" in self.file_callbacks:
            self.file_callbacks["onShowFrame"](frame, source)
        if source == self.selectedPreview:
            self.signals.updatePreviewSignal.emit(frame)

    def onAppSetup(self, app):
        setupFrame = createBlankFrame(500, 500)
        cv2.putText(setupFrame, "Preparing {} app...".format(app.appName), (150, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(setupFrame, "Preparing {} app...".format(app.appName), (150, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        self.signals.updatePreviewSignal.emit(setupFrame)

    def onAppStart(self, app):
        setupFrame = createBlankFrame(500, 500)
        cv2.putText(setupFrame, "Running {} app... (check console)".format(app.appName), (100, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(setupFrame, "Running {} app... (check console)".format(app.appName), (100, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        self.signals.updatePreviewSignal.emit(setupFrame)

    def showDownloadProgress(self, curr, total):
        self.signals.updateDownloadProgressSignal.emit(curr, total)

    def onSetup(self, instance):
        if "onSetup" in self.file_callbacks:
            self.file_callbacks["onSetup"](instance)
        self.signals.updateConfSignal.emit(list(vars(self.conf.args).items()))
        self.signals.setDataSignal.emit(["previewChoices", self.conf.args.show])
        devices = [self.instance._deviceInfo.getMxId()] + list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices()))
        self.signals.setDataSignal.emit(["deviceChoices", devices])
        if instance._nnManager is not None:
            self.signals.setDataSignal.emit(["countLabels", instance._nnManager._labels])
        else:
            self.signals.setDataSignal.emit(["countLabels", []])
        self.signals.setDataSignal.emit(["depthEnabled", self.conf.useDepth])
        self.signals.setDataSignal.emit(["statisticsAccepted", self.instance.metrics is not None])
        self.signals.setDataSignal.emit(["modelChoices", sorted(self.conf.getAvailableZooModels(), key=cmp_to_key(lambda a, b: -1 if a == "mobilenet-ssd" else 1 if b == "mobilenet-ssd" else -1 if a < b else 1))])


class GuiApp(DemoQtGui):
    def __init__(self, instance, args):
        super().__init__()
        self.confManager = prepareConfManager(args)
        self.running = False
        self.selectedPreview = self.confManager.args.show[0] if len(self.confManager.args.show) > 0 else "color"
        self.useDisparity = False
        self.dataInitialized = False
        self.appInitialized = False
        self.threadpool = QThreadPool()
        self._demoInstance = instance

    def updateArg(self, arg_name, arg_value, shouldUpdate=True):
        setattr(self.confManager.args, arg_name, arg_value)
        if shouldUpdate:
            self.worker.signals.setDataSignal.emit(["restartRequired", True])


    def showError(self, error):
        print(error, file=sys.stderr)
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(error)
        msgBox.setWindowTitle("An error occured")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

    def setupDataCollection(self):
        try:
            with Path(".consent").open() as f:
                accepted = json.load(f)["statistics"]
        except:
            accepted = True

        self._demoInstance.toggleMetrics(accepted)

    def start(self):
        self.setupDataCollection()
        self.running = True
        self.worker = Worker(self._demoInstance, parent=self, conf=self.confManager, selectedPreview=self.selectedPreview)
        self.worker.signals.updatePreviewSignal.connect(self.updatePreview)
        self.worker.signals.updateDownloadProgressSignal.connect(self.updateDownloadProgress)
        self.worker.signals.setDataSignal.connect(self.setData)
        self.worker.signals.errorSignal.connect(self.showError)
        self.threadpool.start(self.worker)
        if not self.appInitialized:
            self.appInitialized = True
            exit_code = self.startGui()
            self.stop(wait=False)
            sys.exit(exit_code)

    def stop(self, wait=True):
        if hasattr(self._demoInstance, "_device"):
            current_mxid = self._demoInstance._device.getMxId()
        else:
            current_mxid = self.confManager.args.deviceId
        self.worker.signals.exitSignal.emit()
        self.threadpool.waitForDone(10000)

        if wait and current_mxid is not None:
            start = time.time()
            while time.time() - start < 30:
                if current_mxid in list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices())):
                    break
                else:
                    time.sleep(0.1)
            else:
                print(f"[Warning] Device not available again after 30 seconds! MXID: {current_mxid}")

    def restartDemo(self):
        self.stop()
        self.start()

    def guiOnDepthConfigUpdate(self, median=None, dct=None, sigma=None, lrc=None, lrcThreshold=None):
        self._demoInstance._pm.updateDepthConfig(self._demoInstance._device, median=median, dct=dct, sigma=sigma, lrc=lrc, lrcThreshold=lrcThreshold)
        if median is not None:
            if median == dai.MedianFilter.MEDIAN_OFF:
                self.updateArg("stereoMedianSize", 0, False)
            elif median == dai.MedianFilter.KERNEL_3x3:
                self.updateArg("stereoMedianSize", 3, False)
            elif median == dai.MedianFilter.KERNEL_5x5:
                self.updateArg("stereoMedianSize", 5, False)
            elif median == dai.MedianFilter.KERNEL_7x7:
                self.updateArg("stereoMedianSize", 7, False)
        if dct is not None:
            self.updateArg("disparityConfidenceThreshold", dct, False)
        if sigma is not None:
            self.updateArg("sigma", sigma, False)
        if lrc is not None:
            self.updateArg("stereoLrCheck", lrc, False)
        if lrcThreshold is not None:
            self.updateArg("lrcThreshold", lrcThreshold, False)

    def guiOnCameraConfigUpdate(self, name, exposure=None, sensitivity=None, saturation=None, contrast=None, brightness=None, sharpness=None):
        if exposure is not None:
            newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraExposure or []))) + [(name, exposure)]
            self._demoInstance._cameraConfig["exposure"] = newValue
            self.updateArg("cameraExposure", newValue, False)
        if sensitivity is not None:
            newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraSensitivity or []))) + [(name, sensitivity)]
            self._demoInstance._cameraConfig["sensitivity"] = newValue
            self.updateArg("cameraSensitivity", newValue, False)
        if saturation is not None:
            newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraSaturation or []))) + [(name, saturation)]
            self._demoInstance._cameraConfig["saturation"] = newValue
            self.updateArg("cameraSaturation", newValue, False)
        if contrast is not None:
            newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraContrast or []))) + [(name, contrast)]
            self._demoInstance._cameraConfig["contrast"] = newValue
            self.updateArg("cameraContrast", newValue, False)
        if brightness is not None:
            newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraBrightness or []))) + [(name, brightness)]
            self._demoInstance._cameraConfig["brightness"] = newValue
            self.updateArg("cameraBrightness", newValue, False)
        if sharpness is not None:
            newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraSharpness or []))) + [(name, sharpness)]
            self._demoInstance._cameraConfig["sharpness"] = newValue
            self.updateArg("cameraSharpness", newValue, False)

        self._demoInstance._updateCameraConfigs()

    def guiOnDepthSetupUpdate(self, depthFrom=None, depthTo=None, subpixel=None, extended=None):
        if depthFrom is not None:
            self.updateArg("minDepth", depthFrom)
        if depthTo is not None:
            self.updateArg("maxDepth", depthTo)
        if subpixel is not None:
            self.updateArg("subpixel", subpixel)
        if extended is not None:
            self.updateArg("extendedDisparity", extended)

    def guiOnCameraSetupUpdate(self, name, fps=None, resolution=None):
        if fps is not None:
            if name == "color":
                self.updateArg("rgbFps", fps)
            else:
                self.updateArg("monoFps", fps)
        if resolution is not None:
            if name == "color":
                self.updateArg("rgbResolution", resolution)
            else:
                self.updateArg("monoResolution", resolution)

    def guiOnAiSetupUpdate(self, cnn=None, shave=None, source=None, fullFov=None, sbb=None, sbbFactor=None, ov=None, countLabel=None):
        if cnn is not None:
            self.updateArg("cnnModel", cnn)
        if shave is not None:
            self.updateArg("shaves", shave)
        if source is not None:
            self.updateArg("camera", source)
        if fullFov is not None:
            self.updateArg("disableFullFovNn", not fullFov)
        if sbb is not None:
            self.updateArg("spatialBoundingBox", sbb)
        if sbbFactor is not None:
            self.updateArg("sbbScaleFactor", sbbFactor)
        if ov is not None:
            self.updateArg("openvinoVersion", ov)
        if countLabel is not None or cnn is not None:
            self.updateArg("countLabel", countLabel)

    def guiOnPreviewChangeSelected(self, selected):
        self.worker.selectedPreview = selected
        self.selectedPreview = selected

    def guiOnSelectDevice(self, selected):
        self.updateArg("deviceId", selected)

    def guiOnReloadDevices(self):
        devices = list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices()))
        if hasattr(self._demoInstance, "_deviceInfo"):
            devices.insert(0, self._demoInstance._deviceInfo.getMxId())
        self.worker.signals.setDataSignal.emit(["deviceChoices", devices])
        if len(devices) > 0:
            self.worker.signals.setDataSignal.emit(["restartRequired", True])

    def guiOnStaticticsConsent(self, value):
        try:
            with Path('.consent').open('w') as f:
                json.dump({"statistics": value}, f)
        except:
            pass
        self.worker.signals.setDataSignal.emit(["restartRequired", True])

    def guiOnToggleSync(self, value):
        self.updateArg("sync", value)

    def guiOnToggleColorEncoding(self, enabled, fps):
        oldConfig = self.confManager.args.encode or {}
        if enabled:
            oldConfig["color"] = fps
        elif "color" in self.confManager.args.encode:
            del oldConfig["color"]
        self.updateArg("encode", oldConfig)

    def guiOnToggleLeftEncoding(self, enabled, fps):
        oldConfig = self.confManager.args.encode or {}
        if enabled:
            oldConfig["left"] = fps
        elif "color" in self.confManager.args.encode:
            del oldConfig["left"]
        self.updateArg("encode", oldConfig)

    def guiOnToggleRightEncoding(self, enabled, fps):
        oldConfig = self.confManager.args.encode or {}
        if enabled:
            oldConfig["right"] = fps
        elif "color" in self.confManager.args.encode:
            del oldConfig["right"]
        self.updateArg("encode", oldConfig)

    def guiOnSelectReportingOptions(self, temp, cpu, memory):
        options = []
        if temp:
            options.append("temp")
        if cpu:
            options.append("cpu")
        if memory:
            options.append("memory")
        self.updateArg("report", options)

    def guiOnSelectReportingPath(self, value):
        self.updateArg("reportFile", value)

    def guiOnSelectEncodingPath(self, value):
        self.updateArg("encodeOutput", value)

    def guiOnToggleDepth(self, value):
        self.updateArg("disableDepth", not value)
        selectedPreviews = [Previews.rectifiedRight.name, Previews.rectifiedLeft.name] + ([Previews.disparity.name, Previews.disparityColor.name] if self.useDisparity else [Previews.depth.name, Previews.depthRaw.name])
        depthPreviews = [Previews.rectifiedRight.name, Previews.rectifiedLeft.name, Previews.depth.name, Previews.depthRaw.name, Previews.disparity.name, Previews.disparityColor.name]
        filtered = list(filter(lambda name: name not in depthPreviews, self.confManager.args.show))
        if value:
            updated = filtered + selectedPreviews
            if self.selectedPreview not in updated:
                self.selectedPreview = updated[0]
            self.updateArg("show", updated)
        else:
            updated = filtered + [Previews.left.name, Previews.right.name]
            if self.selectedPreview not in updated:
                self.selectedPreview = updated[0]
            self.updateArg("show", updated)

    def guiOnToggleNN(self, value):
        self.updateArg("disableNeuralNetwork", not value)
        filtered = list(filter(lambda name: name != Previews.nnInput.name, self.confManager.args.show))
        if value:
            updated = filtered + [Previews.nnInput.name]
            if self.selectedPreview not in updated:
                self.selectedPreview = updated[0]
            self.updateArg("show", filtered + [Previews.nnInput.name])
        else:
            if self.selectedPreview not in filtered:
                self.selectedPreview = filtered[0]
            self.updateArg("show", filtered)

    def guiOnRunApp(self, appName):
        self.stop()
        self.updateArg("app", appName, shouldUpdate=False)
        self.setData(["runningApp", appName])
        self.start()

    def guiOnTerminateApp(self, appName):
        self.stop()
        self.updateArg("app", None, shouldUpdate=False)
        self.setData(["runningApp", ""])
        self.start()

    def guiOnToggleDisparity(self, value):
        self.useDisparity = value
        depthPreviews = [Previews.depth.name, Previews.depthRaw.name]
        disparityPreviews = [Previews.disparity.name, Previews.disparityColor.name]
        if value:
            filtered = list(filter(lambda name: name not in depthPreviews, self.confManager.args.show))
            updated = filtered + disparityPreviews
            if self.selectedPreview not in updated:
                self.selectedPreview = updated[0]
            self.updateArg("show", updated)
        else:
            filtered = list(filter(lambda name: name not in disparityPreviews, self.confManager.args.show))
            updated = filtered + depthPreviews
            if self.selectedPreview not in updated:
                self.selectedPreview = updated[0]
            self.updateArg("show", updated)


def runQt(args, demo_instance):
    GuiApp(demo_instance, args).start()

#!/usr/bin/env python3
import atexit
import signal
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import argparse
import json
import os
import time
import traceback
from functools import cmp_to_key
from itertools import cycle
import platform
from pathlib import Path

if platform.machine() == 'aarch64':  # Jetson
    os.environ['OPENBLAS_CORETYPE'] = "ARMV8"

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str((Path(__file__).parent / "depthai_sdk" / "src").absolute()))

try:
    import cv2
    import depthai as dai
    import numpy as np
    from depthai_sdk.managers import ArgsManager, getMonoResolution, getRgbResolution
    from depthai_helpers.app_manager import App
    from depthai_sdk.fps import FPSHandler
    from depthai_sdk.previews import Previews
except Exception as ex:
    print("Third party libraries failed to import: {}".format(ex))
    print("Run \"python3 install_requirements.py\" to install dependencies")
    sys.exit(42)

app = ArgsManager.parseApp()

if __name__ == "__main__":
    if app is not None:
        try:
            app = App(appName=app)
            app.createVenv()
            app.runApp()
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)



from log_system_information import make_sys_report
from depthai_helpers.supervisor import Supervisor
from depthai_helpers.config_manager import ConfigManager, DEPTHAI_ZOO, DEPTHAI_VIDEOS
from depthai_helpers.version_check import checkRequirementsVersion
from depthai_sdk import loadModule, getDeviceInfo, downloadYTVideo, createBlankFrame
from depthai_sdk.managers import NNetManager, SyncedPreviewManager, PreviewManager, PipelineManager, EncodingManager, BlobManager


class OverheatError(RuntimeError):
    pass

args = ArgsManager.parseArgs()

if args.noSupervisor and args.guiType == "qt":
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    if "QT_QPA_FONTDIR" in os.environ:
        os.environ.pop("QT_QPA_FONTDIR")

if not args.noSupervisor:
    print('Using depthai module from: ', dai.__file__)
    print('Depthai version installed: ', dai.__version__)

if not args.debug and not args.skipVersionCheck and platform.machine() not in ['armv6l', 'aarch64']:
    checkRequirementsVersion()

sentryEnabled = False
try:
    import sentry_sdk

    sentry_sdk.init(
        "https://159e328c631a4d3eb0248c0d92e41db3@o1095304.ingest.sentry.io/6114622",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        with_locals=False,
    )
    sentry_sdk.set_context("syslog", make_sys_report(anonymous=True, skipUsb=True, skipPackages=True))
    sentryEnabled = True
except Exception as ex:
    print("Logging and crash reporting disabled! {}".format(ex))

class Trackbars:
    instances = {}

    @staticmethod
    def createTrackbar(name, window, minVal, maxVal, defaultVal, callback):
        def fn(value):
            if Trackbars.instances[name][window] != value:
                callback(value)
            for otherWindow, previousValue in Trackbars.instances[name].items():
                if otherWindow != window and previousValue != value:
                    Trackbars.instances[name][otherWindow] = value
                    cv2.setTrackbarPos(name, otherWindow, value)

        cv2.createTrackbar(name, window, minVal, maxVal, fn)
        Trackbars.instances[name] = {**Trackbars.instances.get(name, {}), window: defaultVal}
        cv2.setTrackbarPos(name, window, defaultVal)


noop = lambda *a, **k: None


class Demo:
    DISP_CONF_MIN = int(os.getenv("DISP_CONF_MIN", 0))
    DISP_CONF_MAX = int(os.getenv("DISP_CONF_MAX", 255))
    SIGMA_MIN = int(os.getenv("SIGMA_MIN", 0))
    SIGMA_MAX = int(os.getenv("SIGMA_MAX", 250))
    LRCT_MIN = int(os.getenv("LRCT_MIN", 0))
    LRCT_MAX = int(os.getenv("LRCT_MAX", 10))
    error = None

    def run_all(self, conf):
        if conf.args.app is not None:
            app = App(appName=conf.args.app)
            self.onAppSetup(app)
            app.createVenv()
            self.onAppStart(app)
            app.runApp(shouldRun=self.shouldRun)
        else:
            self.setup(conf)
            self.run()

    def __init__(self, displayFrames=True, onNewFrame = noop, onShowFrame = noop, onNn = noop, onReport = noop, onSetup = noop, onTeardown = noop, onIter = noop, onAppSetup = noop, onAppStart = noop, shouldRun = lambda: True, showDownloadProgress=None):
        self._openvinoVersion = None
        self._displayFrames = displayFrames

        self.onNewFrame = onNewFrame
        self.onShowFrame = onShowFrame
        self.onNn = onNn
        self.onReport = onReport
        self.onSetup = onSetup
        self.onTeardown = onTeardown
        self.onIter = onIter
        self.shouldRun = shouldRun
        self.showDownloadProgress = showDownloadProgress
        self.onAppSetup = onAppSetup
        self.onAppStart = onAppStart

    def setCallbacks(self, onNewFrame=None, onShowFrame=None, onNn=None, onReport=None, onSetup=None, onTeardown=None, onIter=None, onAppSetup=None, onAppStart=None, shouldRun=None, showDownloadProgress=None):
        if onNewFrame is not None:
            self.onNewFrame = onNewFrame
        if onShowFrame is not None:
            self.onShowFrame = onShowFrame
        if onNn is not None:
            self.onNn = onNn
        if onReport is not None:
            self.onReport = onReport
        if onSetup is not None:
            self.onSetup = onSetup
        if onTeardown is not None:
            self.onTeardown = onTeardown
        if onIter is not None:
            self.onIter = onIter
        if shouldRun is not None:
            self.shouldRun = shouldRun
        if showDownloadProgress is not None:
            self.showDownloadProgress = showDownloadProgress
        if onAppSetup is not None:
            self.onAppSetup = onAppSetup
        if onAppStart is not None:
            self.onAppStart = onAppStart

    def setup(self, conf: ConfigManager):
        print("Setting up demo...")
        self._conf = conf
        if self._conf.args.openvinoVersion:
            self._openvinoVersion = getattr(dai.OpenVINO.Version, 'VERSION_' + self._conf.args.openvinoVersion)
        self._deviceInfo = getDeviceInfo(self._conf.args.deviceId, args.debug)
        if self._conf.args.reportFile:
            reportFileP = Path(self._conf.args.reportFile).with_suffix('.csv')
            reportFileP.parent.mkdir(parents=True, exist_ok=True)
            self._reportFile = reportFileP.open('a')
        self._pm = PipelineManager(openvinoVersion=self._openvinoVersion, lowCapabilities=self._conf.lowCapabilities)

        if self._conf.args.xlinkChunkSize is not None:
            self._pm.setXlinkChunkSize(self._conf.args.xlinkChunkSize)

        if self._conf.args.cameraTuning:
            self._pm.setCameraTuningBlob(self._conf.args.cameraTuning)

        self._nnManager = None
        if self._conf.useNN:
            self._blobManager = BlobManager(
                zooDir=DEPTHAI_ZOO,
                zooName=self._conf.getModelName(),
                progressFunc=self.showDownloadProgress
            )
            self._nnManager = NNetManager(inputSize=self._conf.inputSize, sync=self._conf.args.sync)

            if self._conf.getModelDir() is not None:
                configPath = self._conf.getModelDir() / Path(self._conf.getModelName()).with_suffix(f".json")
                self._nnManager.readConfig(configPath)

            self._nnManager.countLabel(self._conf.getCountLabel(self._nnManager))
            self._pm.setNnManager(self._nnManager)

        maxUsbSpeed = dai.UsbSpeed.HIGH if self._conf.args.usbSpeed == "usb2" else dai.UsbSpeed.SUPER
        self._device = dai.Device(self._pm.pipeline.getOpenVINOVersion(), self._deviceInfo, maxUsbSpeed)
        self._device.addLogCallback(self._logMonitorCallback)
        if sentryEnabled:
            try:
                from sentry_sdk import set_user
                set_user({"mxid": self._device.getMxId()})
            except:
                pass
        if self._deviceInfo.protocol == dai.XLinkProtocol.X_LINK_USB_VSC:
            print("USB Connection speed: {}".format(self._device.getUsbSpeed()))
        self._conf.adjustParamsToDevice(self._device)
        self._conf.adjustPreviewToOptions()
        if self._conf.lowBandwidth:
            self._pm.enableLowBandwidth(poeQuality=self._conf.args.poeQuality)
        self._cap = cv2.VideoCapture(self._conf.args.video) if not self._conf.useCamera else None
        self._fps = FPSHandler() if self._conf.useCamera else FPSHandler(self._cap)
        irDrivers = self._device.getIrDrivers()
        irSetFromCmdLine = any([self._conf.args.irDotBrightness, self._conf.args.irFloodBrightness])
        if irDrivers:
            print('IR drivers found on OAK-D Pro:', [f'{d[0]} on bus {d[1]}' for d in irDrivers])
            if not irSetFromCmdLine: print(' --> Go to the `Depth` tab to enable!')
        elif irSetFromCmdLine:
            print('[ERROR] IR drivers not detected on device!')

        if self._conf.useCamera:
            pvClass = SyncedPreviewManager if self._conf.args.sync else PreviewManager
            self._pv = pvClass(display=self._conf.args.show, nnSource=self._conf.getModelSource(), colorMap=self._conf.getColorMap(),
                               dispMultiplier=self._conf.dispMultiplier, mouseTracker=True, decode=self._conf.lowBandwidth and not self._conf.lowCapabilities,
                               fpsHandler=self._fps, createWindows=self._displayFrames, depthConfig=self._pm._depthConfig)

            if self._conf.leftCameraEnabled:
                self._pm.createLeftCam(args = self._conf.args)
            if self._conf.rightCameraEnabled:
                self._pm.createRightCam(args = self._conf.args)
            if self._conf.rgbCameraEnabled:
                self._pm.createColorCam(args = self._conf.args)

            if self._conf.useDepth:
                self._pm.createDepth(args = self._conf.args)

            if self._conf.irEnabled(self._device):
                self._pm.updateIrConfig(self._device, self._conf.args.irDotBrightness, self._conf.args.irFloodBrightness)

            self._encManager = None
            if len(self._conf.args.encode) > 0:
                self._encManager = EncodingManager(self._conf.args.encode, self._conf.args.encodeOutput)
                self._encManager.createEncoders(self._pm)

        if len(self._conf.args.report) > 0:
            self._pm.createSystemLogger()

        if self._conf.useNN:
            self._nn = self._nnManager.createNN(
                pipeline=self._pm.pipeline, nodes=self._pm.nodes, source=self._conf.getModelSource(),
                blobPath=self._blobManager.getBlob(shaves=self._conf.shaves, openvinoVersion=self._nnManager.openvinoVersion),
                useDepth=self._conf.useDepth, minDepth=self._conf.args.minDepth, maxDepth=self._conf.args.maxDepth,
                sbbScaleFactor=self._conf.args.sbbScaleFactor, fullFov=not self._conf.args.disableFullFovNn,
            )

            self._pm.addNn(nn=self._nn, xoutNnInput=Previews.nnInput.name in self._conf.args.show,
                           xoutSbb=self._conf.args.spatialBoundingBox and self._conf.useDepth)

    def run(self):
        self._device.startPipeline(self._pm.pipeline)
        self._pm.createDefaultQueues(self._device)
        if self._conf.useNN:
            self._nnManager.createQueues(self._device)

        self._sbbOut = self._device.getOutputQueue("sbb", maxSize=1, blocking=False) if self._conf.useNN and self._conf.args.spatialBoundingBox else None
        self._logOut = self._device.getOutputQueue("systemLogger", maxSize=30, blocking=False) if len(self._conf.args.report) > 0 else None

        if self._conf.useDepth:
            self._medianFilters = cycle([item for name, item in vars(dai.MedianFilter).items() if name.startswith('KERNEL_') or name.startswith('MEDIAN_')])
            for medFilter in self._medianFilters:
                # move the cycle to the current median filter
                if medFilter == self._pm._depthConfig.postProcessing.median:
                    break
        else:
            self._medianFilters = []

        if self._conf.useCamera:
            cameras = self._device.getConnectedCameras()
            if dai.CameraBoardSocket.CAM_B in cameras and dai.CameraBoardSocket.CAM_C in cameras:
                self._pv.collectCalibData(self._device)

            self._updateCameraConfigs({
                "exposure": self._conf.args.cameraExposure,
                "sensitivity": self._conf.args.cameraSensitivity,
                "saturation": self._conf.args.cameraSaturation,
                "contrast": self._conf.args.cameraContrast,
                "brightness": self._conf.args.cameraBrightness,
                "sharpness": self._conf.args.cameraSharpness,
            })

            self._pv.createQueues(self._device, self._createQueueCallback)
            if self._encManager is not None:
                self._encManager.createDefaultQueues(self._device)

        self._seqNum = 0
        self._hostFrame = None
        self._nnData = []
        self._sbbRois = []
        self.onSetup(self)

        try:
            while self.shouldRun() and self.canRun():
                self._fps.nextIter()
                self.onIter(self)
                self.loop()
        except StopIteration:
            pass
        except Exception as ex:
            if sentryEnabled:
                from sentry_sdk import capture_exception
                capture_exception(ex)
            raise
        finally:
            self.stop()

    def stop(self, *args, **kwargs):
        if hasattr(self, "_device"):
            print("Stopping demo...")
            self._device.close()
            del self._device
            self._fps.printStatus()
        self._pm.closeDefaultQueues()
        if self._conf.useCamera:
            self._pv.closeQueues()
            if self._encManager is not None:
                self._encManager.close()
        if self._nnManager is not None:
            self._nnManager.closeQueues()
        if self._sbbOut is not None:
            self._sbbOut.close()
        if self._logOut is not None:
            self._logOut.close()
        self.onTeardown(self)

    def canRun(self):
        return hasattr(self, "_device") and not self._device.isClosed()

    def _logMonitorCallback(self, msg):
        if msg.level == dai.LogLevel.CRITICAL:
            print(f"[CRITICAL] [{msg.time.get()}] {msg.payload}", file=sys.stderr)
            sys.stderr.flush()
            temperature = self._device.getChipTemperature()
            if any(map(lambda field: getattr(temperature, field) > 100, ["average", "css", "dss", "mss", "upa"])):
                self.error = OverheatError(msg.payload)
            else:
                self.error = RuntimeError(msg.payload)

    timer = time.monotonic()

    def loop(self):
        diff = time.monotonic() - self.timer
        if diff < 0.02:
            time.sleep(diff)
        self.timer = time.monotonic()

        if self.error is not None:
            self.stop()
            raise self.error

        if self._conf.useCamera:
            self._pv.prepareFrames(callback=self.onNewFrame)
            if self._encManager is not None:
                self._encManager.parseQueues()

            if self._sbbOut is not None:
                sbb = self._sbbOut.tryGet()
                if sbb is not None:
                    self._sbbRois = sbb.getConfigData()
                depthFrames = [self._pv.get(Previews.depthRaw.name), self._pv.get(Previews.depth.name)]
                for depthFrame in depthFrames:
                    if depthFrame is None:
                        continue

                    for roiData in self._sbbRois:
                        roi = roiData.roi.denormalize(depthFrame.shape[1], depthFrame.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        # Display SBB on the disparity map
                        cv2.rectangle(depthFrame, (int(topLeft.x), int(topLeft.y)), (int(bottomRight.x), int(bottomRight.y)), self._nnManager._bboxColors[0], 2)
        else:
            readCorrectly, rawHostFrame = self._cap.read()
            if not readCorrectly:
                raise StopIteration()

            self._nnManager.sendInputFrame(rawHostFrame, self._seqNum)
            self._seqNum += 1
            self._hostFrame = rawHostFrame
            self._fps.tick('host')

        if self._nnManager is not None:
            newData, inNn = self._nnManager.parse()
            if inNn is not None:
                self.onNn(inNn, newData)
                self._fps.tick('nn')
            if newData is not None:
                self._nnData = newData

        if self._conf.useCamera:
            if self._nnManager is not None:
                self._nnManager.draw(self._pv, self._nnData)
            self._pv.showFrames(callback=self._showFramesCallback)
        elif self._hostFrame is not None:
            debugHostFrame = self._hostFrame.copy()
            if self._nnManager is not None:
                self._nnManager.draw(debugHostFrame, self._nnData)
            self._fps.drawFps(debugHostFrame, "host")
            if self._displayFrames:
                cv2.imshow("host", debugHostFrame)

        if self._logOut:
            logs = self._logOut.tryGetAll()
            for log in logs:
                self._printSysInfo(log)

        if self._displayFrames:
            key = cv2.waitKey(1)
            if key == ord('q'):
                raise StopIteration()
            elif key == ord('m'):
                nextFilter = next(self._medianFilters)
                self._pm.updateDepthConfig(self._device, median=nextFilter)

            if self._conf.args.cameraControls:
                update = True

                if key == ord('t'):
                    self._cameraConfig["exposure"] = 10000 if self._cameraConfig["exposure"] is None else 500 if self._cameraConfig["exposure"] == 1 else min(self._cameraConfig["exposure"] + 500, 33000)
                    if self._cameraConfig["sensitivity"] is None:
                        self._cameraConfig["sensitivity"] = 800
                elif key == ord('g'):
                    self._cameraConfig["exposure"] = 10000 if self._cameraConfig["exposure"] is None else max(self._cameraConfig["exposure"] - 500, 1)
                    if self._cameraConfig["sensitivity"] is None:
                        self._cameraConfig["sensitivity"] = 800
                elif key == ord('y'):
                    self._cameraConfig["sensitivity"] = 800 if self._cameraConfig["sensitivity"] is None else min(self._cameraConfig["sensitivity"] + 50, 1600)
                    if self._cameraConfig["exposure"] is None:
                        self._cameraConfig["exposure"] = 10000
                elif key == ord('h'):
                    self._cameraConfig["sensitivity"] = 800 if self._cameraConfig["sensitivity"] is None else max(self._cameraConfig["sensitivity"] - 50, 100)
                    if self._cameraConfig["exposure"] is None:
                        self._cameraConfig["exposure"] = 10000
                elif key == ord('u'):
                    self._cameraConfig["saturation"] = 0 if self._cameraConfig["saturation"] is None else min(self._cameraConfig["saturation"] + 1, 10)
                elif key == ord('j'):
                    self._cameraConfig["saturation"] = 0 if self._cameraConfig["saturation"] is None else max(self._cameraConfig["saturation"] - 1, -10)
                elif key == ord('i'):
                    self._cameraConfig["contrast"] = 0 if self._cameraConfig["contrast"] is None else min(self._cameraConfig["contrast"] + 1, 10)
                elif key == ord('k'):
                    self._cameraConfig["contrast"] = 0 if self._cameraConfig["contrast"] is None else max(self._cameraConfig["contrast"] - 1, -10)
                elif key == ord('o'):
                    self._cameraConfig["brightness"] = 0 if self._cameraConfig["brightness"] is None else min(self._cameraConfig["brightness"] + 1, 10)
                elif key == ord('l'):
                    self._cameraConfig["brightness"] = 0 if self._cameraConfig["brightness"] is None else max(self._cameraConfig["brightness"] - 1, -10)
                elif key == ord('p'):
                    self._cameraConfig["sharpness"] = 0 if self._cameraConfig["sharpness"] is None else min(self._cameraConfig["sharpness"] + 1, 4)
                elif key == ord(';'):
                    self._cameraConfig["sharpness"] = 0 if self._cameraConfig["sharpness"] is None else max(self._cameraConfig["sharpness"] - 1, 0)
                else:
                    update = False

                if update:
                    self._updateCameraConfigs()

    def _createQueueCallback(self, queueName):
        if self._displayFrames and queueName in [Previews.disparityColor.name, Previews.disparity.name, Previews.depth.name, Previews.depthRaw.name]:
            Trackbars.createTrackbar('Disparity confidence', queueName, self.DISP_CONF_MIN, self.DISP_CONF_MAX, self._conf.args.disparityConfidenceThreshold,
                     lambda value: self._pm.updateDepthConfig(dct=value))
            if queueName in [Previews.depthRaw.name, Previews.depth.name]:
                Trackbars.createTrackbar('Bilateral sigma', queueName, self.SIGMA_MIN, self.SIGMA_MAX, self._conf.args.sigma,
                         lambda value: self._pm.updateDepthConfig(sigma=value))
            if self._conf.args.stereoLrCheck:
                Trackbars.createTrackbar('LR-check threshold', queueName, self.LRCT_MIN, self.LRCT_MAX, self._conf.args.lrcThreshold,
                         lambda value: self._pm.updateDepthConfig(lrcThreshold=value))
            if self._device.getIrDrivers():
                Trackbars.createTrackbar('IR Laser Dot Projector [mA]', queueName, 0, 1200, self._conf.args.irDotBrightness,
                         lambda value: self._device.setIrLaserDotProjectorBrightness(value))
                Trackbars.createTrackbar('IR Flood Illuminator [mA]', queueName, 0, 1500, self._conf.args.irFloodBrightness,
                         lambda value: self._device.setIrFloodLightBrightness(value))

    def _updateCameraConfigs(self, config):
        parsedConfig = {}
        for configOption, values in config.items():
            if values is not None:
                for cameraName, value in values:
                    newConfig = {
                        **parsedConfig.get(cameraName, {}),
                        configOption: value
                    }
                    if cameraName == "all":
                        parsedConfig[Previews.left.name] = newConfig
                        parsedConfig[Previews.right.name] = newConfig
                        parsedConfig[Previews.color.name] = newConfig
                    else:
                        parsedConfig[cameraName] = newConfig

        if self._conf.leftCameraEnabled and Previews.left.name in parsedConfig:
            self._pm.updateLeftCamConfig(**parsedConfig[Previews.left.name])
        if self._conf.rightCameraEnabled and Previews.right.name in parsedConfig:
            self._pm.updateRightCamConfig(**parsedConfig[Previews.right.name])
        if self._conf.rgbCameraEnabled and Previews.color.name in parsedConfig:
            self._pm.updateColorCamConfig(**parsedConfig[Previews.color.name])

    def _showFramesCallback(self, frame, name):
        returnFrame = self.onShowFrame(frame, name)
        return returnFrame if returnFrame is not None else frame


    def _printSysInfo(self, info):
        m = 1024 * 1024 # MiB
        if not hasattr(self, "_reportFile"):
            if "memory" in self._conf.args.report:
                print(f"Drr used / total - {info.ddrMemoryUsage.used / m:.2f} / {info.ddrMemoryUsage.total / m:.2f} MiB")
                print(f"Cmx used / total - {info.cmxMemoryUsage.used / m:.2f} / {info.cmxMemoryUsage.total / m:.2f} MiB")
                print(f"LeonCss heap used / total - {info.leonCssMemoryUsage.used / m:.2f} / {info.leonCssMemoryUsage.total / m:.2f} MiB")
                print(f"LeonMss heap used / total - {info.leonMssMemoryUsage.used / m:.2f} / {info.leonMssMemoryUsage.total / m:.2f} MiB")
            if "temp" in self._conf.args.report:
                t = info.chipTemperature
                print(f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa0: {t.upa:.2f}, upa1: {t.dss:.2f}")
            if "cpu" in self._conf.args.report:
                print(f"Cpu usage - Leon OS: {info.leonCssCpuUsage.average * 100:.2f}%, Leon RT: {info.leonMssCpuUsage.average * 100:.2f} %")
            print("----------------------------------------")
        else:
            data = {}
            if "memory" in self._conf.args.report:
                data = {
                    **data,
                    "ddrUsed": info.ddrMemoryUsage.used,
                    "ddrTotal": info.ddrMemoryUsage.total,
                    "cmxUsed": info.cmxMemoryUsage.used,
                    "cmxTotal": info.cmxMemoryUsage.total,
                    "leonCssUsed": info.leonCssMemoryUsage.used,
                    "leonCssTotal": info.leonCssMemoryUsage.total,
                    "leonMssUsed": info.leonMssMemoryUsage.used,
                    "leonMssTotal": info.leonMssMemoryUsage.total,
                }
            if "temp" in self._conf.args.report:
                data = {
                    **data,
                    "tempAvg": info.chipTemperature.average,
                    "tempCss": info.chipTemperature.css,
                    "tempMss": info.chipTemperature.mss,
                    "tempUpa0": info.chipTemperature.upa,
                    "tempUpa1": info.chipTemperature.dss,
                }
            if "cpu" in self._conf.args.report:
                data = {
                    **data,
                    "cpuCssAvg": info.leonCssCpuUsage.average,
                    "cpuMssAvg": info.leonMssCpuUsage.average,
                }

            if self._reportFile.tell() == 0:
                print(','.join(data.keys()), file=self._reportFile)
            self.onReport(data)
            print(','.join(map(str, data.values())), file=self._reportFile)


def prepareConfManager(in_args):
    confManager = ConfigManager(in_args)
    confManager.linuxCheckApplyUsbRules()
    if not confManager.useCamera:
        if str(confManager.args.video).startswith('https'):
            confManager.args.video = str(downloadYTVideo(confManager.args.video, DEPTHAI_VIDEOS))
            print("Youtube video downloaded.")
        if not Path(confManager.args.video).exists():
            raise ValueError("Path {} does not exists!".format(confManager.args.video))
    return confManager


def runQt():
    from gui.main import DemoQtGui
    from PyQt5.QtWidgets import QMessageBox
    from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool


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
                devices = []
                if args.debug:
                    devices = dai.XLinkConnection.getAllConnectedDevices()
                else:
                    devices = dai.Device.getAllAvailableDevices()
                if len(devices) > 0:
                    defaultDevice = next(map(
                        lambda info: info.getMxId(),
                        filter(lambda info: info.protocol == dai.XLinkProtocol.X_LINK_USB_VSC, devices)
                    ), None)
                    if defaultDevice is None:
                        defaultDevice = devices[0].getMxId()
                    self.conf.args.deviceId = defaultDevice
            if Previews.color.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.color.name)
            if self.conf.useNN and Previews.nnInput.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.nnInput.name)
            if self.conf.useDepth and not self.parent.useDisparity and Previews.depth.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.depth.name)
            if self.conf.useDepth and not self.parent.useDisparity and Previews.depthRaw.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.depthRaw.name)
            if self.conf.useDepth and self.parent.useDisparity and Previews.disparity.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.disparity.name)
            if self.conf.useDepth and self.parent.useDisparity and Previews.disparityColor.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.disparityColor.name)
            if Previews.left.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.left.name)
            if self.conf.useDepth and Previews.rectifiedLeft.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.rectifiedLeft.name)
            if Previews.right.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.right.name)
            if self.conf.useDepth and Previews.rectifiedRight.name not in self.conf.args.show:
                self.conf.args.show.append(Previews.rectifiedRight.name)
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
            self.signals.errorSignal.emit(''.join(traceback.format_tb(ex.__traceback__) + [f"{type(ex).__name__}: {ex}"]))
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
            devices = []
            if args.debug:
                devices = [self.instance._deviceInfo.getMxId()] + list(map(lambda info: info.getMxId(), dai.XLinkConnection.getAllConnectedDevices()))
            else:
                devices = [self.instance._deviceInfo.getMxId()] + list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices()))
            self.signals.setDataSignal.emit(["deviceChoices", devices])
            if instance._nnManager is not None:
                self.signals.setDataSignal.emit(["countLabels", instance._nnManager._labels])
            else:
                self.signals.setDataSignal.emit(["countLabels", []])
            self.signals.setDataSignal.emit(["depthEnabled", self.conf.useDepth])
            self.signals.setDataSignal.emit(["irEnabled", self.conf.irEnabled(instance._device)])
            self.signals.setDataSignal.emit(["irDotBrightness", self.conf.args.irDotBrightness if self.conf.irEnabled(instance._device) else 0])
            self.signals.setDataSignal.emit(["irFloodBrightness", self.conf.args.irFloodBrightness if self.conf.irEnabled(instance._device) else 0])
            self.signals.setDataSignal.emit(["lrc", self.conf.args.stereoLrCheck])
            self.signals.setDataSignal.emit(["modelChoices", sorted(self.conf.getAvailableZooModels(), key=cmp_to_key(lambda a, b: -1 if a == "mobilenet-ssd" else 1 if b == "mobilenet-ssd" else -1 if a < b else 1))])


    class GuiApp(DemoQtGui):
        def __init__(self):
            super().__init__()
            self.confManager = prepareConfManager(args)
            self.running = False
            self.selectedPreview = self.confManager.args.show[0] if len(self.confManager.args.show) > 0 else "color"
            self.useDisparity = False
            self.dataInitialized = False
            self.appInitialized = False
            self.threadpool = QThreadPool()
            self._demoInstance = Demo(displayFrames=False)

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

        def start(self):
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
                try:
                    current_mxid = self._demoInstance._device.getMxId()
                except:
                    current_mxid = self.confManager.args.deviceId
                    del self._demoInstance._device
            else:
                current_mxid = self.confManager.args.deviceId
            self.worker.running = False
            self.worker.signals.exitSignal.emit()
            self.threadpool.waitForDone(10000)

            if wait and current_mxid is not None:
                start = time.time()
                while time.time() - start < 30:
                    localDevices = []
                    if args.debug:
                        localDevices = list(map(lambda info: info.getMxId(), dai.XLinkConnection.getAllConnectedDevices()))
                    else:
                        localDevices = list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices()))

                    if current_mxid in localDevices:
                        break
                    else:
                        time.sleep(0.1)
                else:
                    print(f"[Warning] Device not available again after 30 seconds! MXID: {current_mxid}")

        def restartDemo(self):
            self.stop()
            self.start()

        def stopGui(self, *args, **kwargs):
            self.stop(wait=False)
            self.app.quit()

        def guiOnDepthConfigUpdate(self, median=None, dct=None, sigma=None, lrcThreshold=None, irLaser=None, irFlood=None):
            self._demoInstance._pm.updateDepthConfig(median=median, dct=dct, sigma=sigma, lrcThreshold=lrcThreshold)
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
            if lrcThreshold is not None:
                self.updateArg("lrcThreshold", lrcThreshold, False)
            if any([irLaser, irFlood]):
                self._demoInstance._pm.updateIrConfig(self._demoInstance._device, irLaser, irFlood)
                if irLaser is not None:
                    self.updateArg("irDotBrightness", irLaser, False)
                if irFlood is not None:
                    self.updateArg("irFloodBrightness", irFlood, False)

        def guiOnCameraConfigUpdate(self, name, exposure=None, sensitivity=None, saturation=None, contrast=None, brightness=None, sharpness=None):
            print(name)
            config = {}
            if exposure is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraExposure or []))) + [(name, exposure)]
                config["exposure"] = newValue
                self.updateArg("cameraExposure", newValue, False)
            if sensitivity is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraSensitivity or []))) + [(name, sensitivity)]
                config["sensitivity"] = newValue
                self.updateArg("cameraSensitivity", newValue, False)
            if saturation is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraSaturation or []))) + [(name, saturation)]
                config["saturation"] = newValue
                self.updateArg("cameraSaturation", newValue, False)
            if contrast is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraContrast or []))) + [(name, contrast)]
                config["contrast"] = newValue
                self.updateArg("cameraContrast", newValue, False)
            if brightness is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraBrightness or []))) + [(name, brightness)]
                config["brightness"] = newValue
                self.updateArg("cameraBrightness", newValue, False)
            if sharpness is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.confManager.args.cameraSharpness or []))) + [(name, sharpness)]
                config["sharpness"] = newValue
                self.updateArg("cameraSharpness", newValue, False)

            self._demoInstance._updateCameraConfigs(config)

        def guiOnDepthSetupUpdate(self, depthFrom=None, depthTo=None, subpixel=None, extended=None, lrc=None):
            if depthFrom is not None:
                self.updateArg("minDepth", depthFrom)
            if depthTo is not None:
                self.updateArg("maxDepth", depthTo)
            if subpixel is not None:
                self.updateArg("subpixel", subpixel)
            if extended is not None:
                self.updateArg("extendedDisparity", extended)
            if lrc is not None:
                self.updateArg("stereoLrCheck", lrc)

        def guiOnCameraSetupUpdate(self, name, fps=None, resolution=None):
            if fps is not None:
                if name == "color":
                    self.updateArg("rgbFps", fps)
                else:
                    self.updateArg("monoFps", fps)
            if resolution is not None:
                if name == "color":
                    res = getRgbResolution(resolution)
                    self.updateArg("rgbResolution", res)
                    # Not ideal, we need to refactor this (throw the whole SDK away)
                    self.updateArg("rgbResWidth", self.confManager.rgbResolutionWidth(res))
                else:
                    self.updateArg("monoResolution", getMonoResolution(resolution))

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
            devices = []
            if args.debug:
                devices = list(map(lambda info: info.getMxId(), dai.XLinkConnection.getAllConnectedDevices()))
            else:
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

        def guiOnToggleRgbDepthAlignment(self, value):
            self.updateArg("noRgbDepthAlign", value)

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
                updated = filtered
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
    app = GuiApp()
    signal.signal(signal.SIGINT, app.stopGui)
    signal.signal(signal.SIGTERM, app.stopGui)
    atexit.register(app.stopGui)
    app.start()


def runOpenCv():
    confManager = prepareConfManager(args)
    demo = Demo()
    signal.signal(signal.SIGINT, demo.stop)
    signal.signal(signal.SIGTERM, demo.stop)
    atexit.register(demo.stop)
    demo.run_all(confManager)


if __name__ == "__main__":
    try:
        if args.noSupervisor:
            if args.guiType == "qt":
                runQt()
            else:
                args.guiType = "cv"
                runOpenCv()
        else:
            s = Supervisor()
            if args.guiType != "cv":
                available = s.checkQtAvailability()
                if args.guiType == "qt" and not available:
                    raise RuntimeError("QT backend is not available, run the script with --guiType \"cv\" to use OpenCV backend")
                if args.guiType == "auto" and platform.machine() == 'aarch64':  # Disable Qt by default on Jetson due to Qt issues
                    args.guiType = "cv"
                elif available:
                    args.guiType = "qt"
                else:
                    args.guiType = "cv"
            s.runDemo(args)
    except KeyboardInterrupt:
        sys.exit(0)

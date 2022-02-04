#!/usr/bin/env python3
import sys
import time

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import os
from itertools import cycle
from pathlib import Path
import platform

if platform.machine() == 'aarch64':  # Jetson
    os.environ['OPENBLAS_CORETYPE'] = "ARMV8"

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str((Path(__file__).parent / "depthai_sdk" / "src").absolute()))

from depthai_helpers.app_manager import App
if __name__ == "__main__":
    if '--app' in sys.argv:
        try:
            app = App(appName=sys.argv[sys.argv.index('--app') + 1])
            app.createVenv()
            app.runApp()
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)

try:
    import cv2
    import depthai as dai
    import numpy as np
except Exception as ex:
    print("Third party libraries failed to import: {}".format(ex))
    print("Run \"python3 install_requirements.py\" to install dependencies or visit our installation page for more details - https://docs.luxonis.com/projects/api/en/latest/install/")
    sys.exit(42)

from log_system_information import make_sys_report
from depthai_helpers.supervisor import Supervisor
from depthai_helpers.arg_manager import parseArgs
from depthai_helpers.config_manager import ConfigManager, DEPTHAI_ZOO, DEPTHAI_VIDEOS, prepareConfManager
from depthai_helpers.metrics import MetricManager
from depthai_helpers.version_check import checkRequirementsVersion
from depthai_sdk import FPSHandler, loadModule, getDeviceInfo, downloadYTVideo, Previews, createBlankFrame
from depthai_sdk.managers import NNetManager, SyncedPreviewManager, PreviewManager, PipelineManager, EncodingManager, BlobManager

args = parseArgs()

if args.noSupervisor and args.guiType == "qt":
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    if "QT_QPA_FONTDIR" in os.environ:
        os.environ.pop("QT_QPA_FONTDIR")

if not args.noSupervisor:
    print('Using depthai module from: ', dai.__file__)
    print('Depthai version installed: ', dai.__version__)

if not args.skipVersionCheck and platform.machine() not in ['armv6l', 'aarch64']:
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

    def __init__(self, displayFrames=True, consumeFrames=True, onNewFrame = noop, onShowFrame = noop, onNn = noop, onReport = noop, onPipeline = noop, onSetup = noop, onTeardown = noop, onIter = noop, onAppSetup = noop, onAppStart = noop, shouldRun = lambda: True, showDownloadProgress=None, collectMetrics=False):
        self._openvinoVersion = None
        self._displayFrames = displayFrames
        self._consumeFrames = consumeFrames
        self.toggleMetrics(collectMetrics)

        self.onNewFrame = onNewFrame
        self.onShowFrame = onShowFrame
        self.onNn = onNn
        self.onReport = onReport
        self.onSetup = onSetup
        self.onPipeline = onPipeline
        self.onTeardown = onTeardown
        self.onIter = onIter
        self.shouldRun = shouldRun
        self.showDownloadProgress = showDownloadProgress
        self.onAppSetup = onAppSetup
        self.onAppStart = onAppStart

    def setCallbacks(self, onNewFrame=None, onShowFrame=None, onNn=None, onReport=None, onPipeline=None, onSetup=None, onTeardown=None, onIter=None, onAppSetup=None, onAppStart=None, shouldRun=None, showDownloadProgress=None):
        if onNewFrame is not None:
            self.onNewFrame = onNewFrame
        if onShowFrame is not None:
            self.onShowFrame = onShowFrame
        if onNn is not None:
            self.onNn = onNn
        if onReport is not None:
            self.onReport = onReport
        if onPipeline is not None:
            self.onPipeline = onPipeline
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

    def toggleMetrics(self, enabled):
        if enabled:
            self.metrics = MetricManager()
        else:
            self.metrics = None

    def setup(self, conf: ConfigManager):
        print("Setting up demo...")
        self._conf = conf
        self._rgbRes = conf.getRgbResolution()
        self._monoRes = conf.getMonoResolution()
        if self._conf.args.openvinoVersion:
            self._openvinoVersion = getattr(dai.OpenVINO.Version, 'VERSION_' + self._conf.args.openvinoVersion)
        self._deviceInfo = getDeviceInfo(self._conf.args.deviceId)
        if self._conf.args.reportFile:
            reportFileP = Path(self._conf.args.reportFile).with_suffix('.csv')
            reportFileP.parent.mkdir(parents=True, exist_ok=True)
            self._reportFile = reportFileP.open('a')
        self._pm = PipelineManager(openvinoVersion=self._openvinoVersion, lowCapabilities=self._conf.lowCapabilities)

        if self._conf.args.xlinkChunkSize is not None:
            self._pm.setXlinkChunkSize(self._conf.args.xlinkChunkSize)

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

        self._device = dai.Device(self._pm.pipeline.getOpenVINOVersion(), self._deviceInfo, usb2Mode=self._conf.args.usbSpeed == "usb2")
        if sentryEnabled:
            try:
                from sentry_sdk import set_user
                set_user({"mxid": self._device.getMxId()})
            except:
                pass
        if self.metrics is not None:
            self.metrics.reportDevice(self._device)
        if self._deviceInfo.desc.protocol == dai.XLinkProtocol.X_LINK_USB_VSC:
            print("USB Connection speed: {}".format(self._device.getUsbSpeed()))
        self._conf.adjustParamsToDevice(self._device)
        self._conf.adjustPreviewToOptions()
        if self._conf.lowBandwidth:
            self._pm.enableLowBandwidth(poeQuality=self._conf.args.poeQuality)
        self._cap = cv2.VideoCapture(self._conf.args.video) if not self._conf.useCamera else None
        self._fps = FPSHandler() if self._conf.useCamera else FPSHandler(self._cap)

        if self._conf.useCamera:
            pvClass = SyncedPreviewManager if self._conf.args.sync else PreviewManager
            self._pv = pvClass(display=self._conf.args.show, nnSource=self._conf.getModelSource(), colorMap=self._conf.getColorMap(),
                               dispMultiplier=self._conf.dispMultiplier, mouseTracker=True, decode=self._conf.lowBandwidth and not self._conf.lowCapabilities,
                               fpsHandler=self._fps, createWindows=self._displayFrames, depthConfig=self._pm._depthConfig)

            if self._conf.leftCameraEnabled:
                self._pm.createLeftCam(self._monoRes, self._conf.args.monoFps,
                                 orientation=self._conf.args.cameraOrientation.get(Previews.left.name),
                                 xout=Previews.left.name in self._conf.args.show and self._consumeFrames)
            if self._conf.rightCameraEnabled:
                self._pm.createRightCam(self._monoRes, self._conf.args.monoFps,
                                  orientation=self._conf.args.cameraOrientation.get(Previews.right.name),
                                  xout=Previews.right.name in self._conf.args.show and self._consumeFrames)
            if self._conf.rgbCameraEnabled:
                self._pm.createColorCam(previewSize=self._conf.previewSize, res=self._rgbRes, fps=self._conf.args.rgbFps,
                                  orientation=self._conf.args.cameraOrientation.get(Previews.color.name),
                                  fullFov=not self._conf.args.disableFullFovNn,
                                  xout=Previews.color.name in self._conf.args.show and self._consumeFrames)

            if self._conf.useDepth:
                self._pm.createDepth(
                    self._conf.args.disparityConfidenceThreshold,
                    self._conf.getMedianFilter(),
                    self._conf.args.sigma,
                    self._conf.args.stereoLrCheck,
                    self._conf.args.lrcThreshold,
                    self._conf.args.extendedDisparity,
                    self._conf.args.subpixel,
                    useDepth=Previews.depth.name in self._conf.args.show or Previews.depthRaw.name in self._conf.args.show and self._consumeFrames,
                    useDisparity=Previews.disparity.name in self._conf.args.show or Previews.disparityColor.name in self._conf.args.show and self._consumeFrames,
                    useRectifiedLeft=Previews.rectifiedLeft.name in self._conf.args.show and self._consumeFrames,
                    useRectifiedRight=Previews.rectifiedRight.name in self._conf.args.show and self._consumeFrames,
                )

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

            self._pm.addNn(nn=self._nn, xoutNnInput=Previews.nnInput.name in self._conf.args.show and self._consumeFrames,
                           xoutSbb=self._conf.args.spatialBoundingBox and self._conf.useDepth)

    def run(self):
        self.onPipeline(self._pm.pipeline, self._pm.nodes)
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
            if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                self._pv.collectCalibData(self._device)

            self._cameraConfig = {
                "exposure": self._conf.args.cameraExposure,
                "sensitivity": self._conf.args.cameraSensitivity,
                "saturation": self._conf.args.cameraSaturation,
                "contrast": self._conf.args.cameraContrast,
                "brightness": self._conf.args.cameraBrightness,
                "sharpness": self._conf.args.cameraSharpness
            }

            if any(self._cameraConfig.values()):
                self._updateCameraConfigs()

            if self._consumeFrames:
                self._pv.createQueues(self._device, self._createQueueCallback)
            if self._encManager is not None:
                self._encManager.createDefaultQueues(self._device)

        self._seqNum = 0
        self._hostFrame = None
        self._nnData = []
        self._sbbRois = []
        self.onSetup(self)

        try:
            while not self._device.isClosed() and self.shouldRun():
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

    def stop(self):
        print("Stopping demo...")
        self._device.close()
        del self._device
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
        self._fps.printStatus()
        self.onTeardown(self)

    timer = time.monotonic()

    def loop(self):
        diff = time.monotonic() - self.timer
        if diff < 0.02:
            time.sleep(diff)
        self.timer = time.monotonic()

        if self._conf.useCamera:
            if self._consumeFrames:
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

            if self._conf.args.cameraControlls:
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
                     lambda value: self._pm.updateDepthConfig(self._device, dct=value))
            if queueName in [Previews.depthRaw.name, Previews.depth.name]:
                Trackbars.createTrackbar('Bilateral sigma', queueName, self.SIGMA_MIN, self.SIGMA_MAX, self._conf.args.sigma,
                         lambda value: self._pm.updateDepthConfig(self._device, sigma=value))
            if self._conf.args.stereoLrCheck:
                Trackbars.createTrackbar('LR-check threshold', queueName, self.LRCT_MIN, self.LRCT_MAX, self._conf.args.lrcThreshold,
                         lambda value: self._pm.updateDepthConfig(self._device, lrcThreshold=value))

    def _updateCameraConfigs(self):
        parsedConfig = {}
        for configOption, values in self._cameraConfig.items():
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

        if hasattr(self, "_device"):
            if self._conf.leftCameraEnabled and Previews.left.name in parsedConfig:
                self._pm.updateLeftCamConfig(self._device, **parsedConfig[Previews.left.name])
            if self._conf.rightCameraEnabled and Previews.right.name in parsedConfig:
                self._pm.updateRightCamConfig(self._device, **parsedConfig[Previews.right.name])
            if self._conf.rgbCameraEnabled and Previews.color.name in parsedConfig:
                self._pm.updateColorCamConfig(self._device, **parsedConfig[Previews.color.name])

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


def runOpenCv(in_args, instance):
    confManager = prepareConfManager(in_args)
    instance.run_all(confManager)


if __name__ == "__main__":
    try:
        if args.noSupervisor:
            if args.guiType == "qt":
                from gui.qt.main import runQt
                runQt(args, Demo(displayFrames=False))
            elif args.guiType == "web":
                from gui.web.main import runWeb
                runWeb(args, Demo(displayFrames=False, consumeFrames=False))
            else:
                args.guiType = "cv"
                runOpenCv(args, Demo(displayFrames=True))
        else:
            s = Supervisor()
            if args.guiType in ("auto", "qt"):
                available = s.checkQtAvailability()
                if args.guiType == "qt" and not available:
                    raise RuntimeError("QT backend is not available, run the script with --guiType \"cv\" to use OpenCV backend")
                if available:
                    args.guiType = "qt"
            if args.guiType in ("auto", "cv"):
                if args.guiType == "auto" and platform.machine() == 'aarch64':  # Disable Qt by default on Jetson due to Qt issues
                    args.guiType = "cv"
                args.guiType = "cv"
            if args.guiType in ("auto", "web"):
                args.guiType = "web"
            s.runDemo(args)
    except KeyboardInterrupt:
        sys.exit(0)

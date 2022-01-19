# This Python file uses the following encoding: utf-8
import time
from functools import cmp_to_key

import cv2
import depthai as dai
from depthai_sdk import createBlankFrame

from depthai_helpers.config_manager import prepareConfManager

class WebApp:
    def __init__(self, instance, args):
        super().__init__()
        self.confManager = prepareConfManager(args)
        self.running = False
        self.selectedPreview = self.confManager.args.show[0] if len(self.confManager.args.show) > 0 else "color"
        self._demoInstance = instance
        self._demoInstance.setCallbacks(shouldRun=self.shouldRun, onShowFrame=self.onShowFrame, onSetup=self.onSetup, onAppSetup=self.onAppSetup, onAppStart=self.onAppStart, showDownloadProgress=self.showDownloadProgress)

    def shouldRun(self):
        return True

    def onShowFrame(self, frame, source):
        if source == self.selectedPreview:
            print(frame, source)

    def onSetup(self, instance):
        previewChoices = self.confManager.args.show
        devices = [instance._deviceInfo.getMxId()] + list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices()))
        countLabels = instance._nnManager._labels if instance._nnManager is not None else []
        depthEnabled = self.confManager.useDepth
        modelChoices = sorted(self.confManager.getAvailableZooModels(), key=cmp_to_key(lambda a, b: -1 if a == "mobilenet-ssd" else 1 if b == "mobilenet-ssd" else -1 if a < b else 1))


    def onAppSetup(self, app):
        setupFrame = createBlankFrame(500, 500)
        cv2.putText(setupFrame, "Preparing {} app...".format(app.appName), (150, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(setupFrame, "Preparing {} app...".format(app.appName), (150, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        print(setupFrame)

    def onAppStart(self, app):
        setupFrame = createBlankFrame(500, 500)
        cv2.putText(setupFrame, "Running {} app... (check console)".format(app.appName), (100, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(setupFrame, "Running {} app... (check console)".format(app.appName), (100, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        print(setupFrame)

    def showDownloadProgress(self, curr, total):
        print(curr, total)

    def start(self):
        self.running = True
        print("Starting...")

    def stop(self, wait=True):
        if hasattr(self._demoInstance, "_device"):
            current_mxid = self._demoInstance._device.getMxId()
        else:
            current_mxid = self.confManager.args.deviceId

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


def runWeb(args, demo_instance):
    WebApp(demo_instance, args).start()

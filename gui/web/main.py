# This Python file uses the following encoding: utf-8
import json
import sys
import threading
import time
import traceback
from functools import cmp_to_key
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from pathlib import Path

from PIL import Image
import cv2
import depthai as dai
from depthai_sdk import createBlankFrame, Previews

from depthai_helpers.config_manager import prepareConfManager


class HttpHandler(SimpleHTTPRequestHandler):
    static_path = "./dist"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str((Path(__file__).parent / self.static_path).absolute()), **kwargs)

    def setup(self):
        super().setup()
        self.routes = {
            "/stream": self.stream,
            "/config": self.config,
        }

    # def translate_path(self, path):
    #     if path.startswith(self.static_prefix):
    #         return super().translate_path(path[len(self.static_prefix):])
    #     else:
    #         return super().translate_path("/404.html")

    def do_GET(self):
        if self.path in self.routes.keys():
            return self.routes[self.path]()
        else:
            return super().do_GET()

    def config(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(getattr(self.server, 'config', {})).encode('UTF-8'))

    def stream(self):
        try:
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                if hasattr(self.server, 'frametosend'):
                    image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                    stream_file = BytesIO()
                    image.save(stream_file, 'JPEG')
                    self.wfile.write("--jpgboundary".encode())

                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                    self.end_headers()
                    image.save(self.wfile, 'JPEG')
        except BrokenPipeError:
            return


class WebApp:
    def __init__(self, instance, args):
        super().__init__()
        self.confManager = prepareConfManager(args)
        self.running = False
        self.webserver = None
        self.selectedPreview = self.confManager.args.show[0] if len(self.confManager.args.show) > 0 else "color"
        self._demoInstance = instance
        self.thread = None

    def shouldRun(self):
        return True

    def onShowFrame(self, frame, source):
        if source == self.selectedPreview:
            self.webserver.frametosend = frame

    def onSetup(self, instance):
        previewChoices = self.confManager.args.show
        devices = [instance._deviceInfo.getMxId()] + list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices()))
        countLabels = instance._nnManager._labels if instance._nnManager is not None else []
        depthEnabled = self.confManager.useDepth
        modelChoices = sorted(self.confManager.getAvailableZooModels(), key=cmp_to_key(lambda a, b: -1 if a == "mobilenet-ssd" else 1 if b == "mobilenet-ssd" else -1 if a < b else 1))

        self.webserver.config = {
            "previewChoices": previewChoices,
            "devices": devices,
            "countLabels": countLabels,
            "depthEnabled": depthEnabled,
            "modelChoices": modelChoices,
        }


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

    def onError(self, ex: Exception):
        exception_message = ''.join(traceback.format_tb(ex.__traceback__) + [str(ex)])
        print(exception_message)

    def runDemo(self):
        self._demoInstance.setCallbacks(
            shouldRun=self.shouldRun, onShowFrame=self.onShowFrame, onSetup=self.onSetup, onAppSetup=self.onAppSetup,
            onAppStart=self.onAppStart, showDownloadProgress=self.showDownloadProgress
        )
        self.confManager.args.bandwidth = "auto"
        if self.confManager.args.deviceId is None:
            devices = dai.Device.getAllAvailableDevices()
            if len(devices) > 0:
                defaultDevice = next(map(
                    lambda info: info.getMxId(),
                    filter(lambda info: info.desc.protocol == dai.XLinkProtocol.X_LINK_USB_VSC, devices)
                ), None)
                if defaultDevice is None:
                    defaultDevice = devices[0].getMxId()
                self.confManager.args.deviceId = defaultDevice
        self.confManager.args.show = [
            Previews.color.name, Previews.nnInput.name, Previews.depth.name, Previews.depthRaw.name, Previews.left.name,
            Previews.rectifiedLeft.name, Previews.right.name, Previews.rectifiedRight.name
        ]
        try:
            self._demoInstance.run_all(self.confManager)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as ex:
            self.onError(ex)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.runDemo)
        self.thread.daemon = True
        self.thread.start()

        if self.webserver is None:
            self.webserver = HTTPServer((self.confManager.args.host, self.confManager.args.port), HttpHandler)
            print("Server started http://{}:{}".format(self.confManager.args.host, self.confManager.args.port))

            try:
                self.webserver.serve_forever()
            except KeyboardInterrupt:
                pass

            self.webserver.server_close()

    def stop(self, wait=True):
        if hasattr(self._demoInstance, "_device"):
            current_mxid = self._demoInstance._device.getMxId()
        else:
            current_mxid = self.confManager.args.deviceId

        self.running = False
        self.thread.join()

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

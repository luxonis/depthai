# This Python file uses the following encoding: utf-8
import json
import mimetypes
import sys
import threading
import time
import traceback
from functools import cmp_to_key
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path

from PIL import Image
import cv2
import depthai as dai
from depthai_sdk import createBlankFrame, Previews
from depthai_helpers.arg_manager import openvinoVersions, colorMaps, streamChoices, cameraChoices, reportingChoices, \
    projectRoot
from depthai_helpers.config_manager import prepareConfManager


class HttpHandler(BaseHTTPRequestHandler):
    static_path = Path(__file__).parent / "dist"

    def setup(self):
        super().setup()
        self.routes = {
            "/stream": self.stream,
            "/config": self.config,
            "/update": self.update,
            "/updatePreview": self.updatePreview,
        }

    def do_GET(self):
        if self.path in self.routes.keys():
            return self.routes[self.path]()
        else:
            filePath = self.static_path / self.path.lstrip("/")
            if filePath.is_dir():
                filePath = filePath / "index.html"
            elif not filePath.exists():
                filePath = filePath.with_suffix(".html")
            print(filePath, self.static_path, self.path.lstrip("/"), self.static_path / self.path.lstrip("/"))

            if filePath.exists():
                self.send_response(200)
                mimetype, _ = mimetypes.guess_type(filePath)
                self.send_header('Content-type', mimetype)
                self.end_headers()
                with filePath.open('rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

    def do_POST(self):
        if self.path in self.routes.keys():
            return self.routes[self.path]()
        else:
            self.send_response(404)
            self.end_headers()

    def config(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(self.server.config).encode('UTF-8'))

    def updatePreview(self):
        if self.server.instance is None:
            self.send_response(202)
            self.end_headers()
            return

        post_body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        test_data = json.loads(post_body)
        self.server.instance.selectedPreview = test_data["preview"]
        self.send_response(200)
        self.end_headers()

    def update(self):
        if self.server.instance is None:
            self.send_response(202)
            self.end_headers()
            return

        post_body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        test_data = json.loads(post_body)

        def updatePreview(data):
            self.server.instance.selectedPreview = data

        def updateStatistics(data):
            try:
                with Path(projectRoot / ".consent").open('w') as f:
                    json.dump({"statistics": data}, f)
            except:
                pass

        mapping = {
            "ai": {
                "enabled": lambda data: self.server.instance.updateArg("disableNeuralNetwork", not data),
                "model": lambda data: self.server.instance.updateArg("cnnModel", data["current"]) if "current" in data else None,
                "fullFov": lambda data: self.server.instance.updateArg("disableFullFovNn", not data),
                "source": lambda data: self.server.instance.updateArg("camera", data["current"]),
                "shaves": lambda data: self.server.instance.updateArg("shaves", data),
                "ovVersion": lambda data: self.server.instance.updateArg("openvinoVersion", data["current"]) if "current" in data else None,
                "label": lambda data: self.server.instance.updateArg("countLabel", data["current"]) if "current" in data else None,
                "sbb": lambda data: self.server.instance.updateArg("spatialBoundingBox", data),
                "sbbFactor": lambda data: self.server.instance.updateArg("sbbScaleFactor", data),
            },
            "depth": {
                "enabled": lambda data: self.server.instance.updateArg("disableDepth", not data),
                "median": lambda data: self.server.instance.updateArg("stereoMedianSize", int(data["current"].replace("KERNEL_", "").split("x")[0]) if data["current"].startswith("KERNEL_") else 0) if "current" in data else None,
                "subpixel": lambda data: self.server.instance.updateArg("subpixel", data),
                "lrc": lambda data: self.server.instance.updateArg("stereoLrCheck", data),
                "extended": lambda data: self.server.instance.updateArg("extendedDisparity", data),
                "confidence": lambda data: self.server.instance.updateArg("disparityConfidenceThreshold", data),
                "sigma": lambda data: self.server.instance.updateArg("sigma", data),
                "lrcThreshold": lambda data: self.server.instance.updateArg("lrcThreshold", data),
                "range": {
                    "min": lambda data: self.server.instance.updateArg("minDepth", data),
                    "max": lambda data: self.server.instance.updateArg("maxDepth", data),
                },
            },
            "camera": {
                "sync": lambda data: self.server.instance.updateArg("sync", data),
                "color": {
                    "fps": lambda data: self.server.instance.updateArg("rgbFps", data),
                    "resolution": lambda data: self.server.instance.updateArg("rgbResolution", data["current"]) if "current" in data else None,
                    "iso": lambda data: self.server.instance.updateArg("cameraSensitivity", data),
                    "exposure": lambda data: self.server.instance.updateArg("cameraExposure", data),
                    "saturation": lambda data: self.server.instance.updateArg("cameraSaturation", data),
                    "contrast": lambda data: self.server.instance.updateArg("cameraContrast", data),
                    "brightness": lambda data: self.server.instance.updateArg("cameraBrightness", data),
                    "sharpness": lambda data: self.server.instance.updateArg("cameraSharpness", data),
                },
                "mono": {
                    "fps": lambda data: self.server.instance.updateArg("monoFps", data),
                    "resolution": lambda data: self.server.instance.updateArg("monoResolution", data["current"]) if "current" in data else None,
                    "iso": lambda data: self.server.instance.updateArg("cameraSensitivity", data),
                    "exposure": lambda data: self.server.instance.updateArg("cameraExposure", data),
                    "saturation": lambda data: self.server.instance.updateArg("cameraSaturation", data),
                    "contrast": lambda data: self.server.instance.updateArg("cameraContrast", data),
                    "brightness": lambda data: self.server.instance.updateArg("cameraBrightness", data),
                    "sharpness": lambda data: self.server.instance.updateArg("cameraSharpness", data),
                }
            },
            "misc": {
                "recording": {
                    "color": lambda data: self.server.instance.updateArg("encode", {**self.confManager.args.encode, "color": data}),
                    "left": lambda data: self.server.instance.updateArg("left", {**self.confManager.args.encode, "left": data}),
                    "right": lambda data: self.server.instance.updateArg("right", {**self.confManager.args.encode, "right": data}),
                    "dest": lambda data: self.server.instance.updateArg("encodeOutput", data),
                },
                "reporting": {
                    "enabled": lambda data: self.server.instance.updateArg("report", data),
                    "dest": lambda data: self.server.instance.updateArg("reportFile", data),
                },
                "demo": {
                    "statistics": updateStatistics,
                }
            },
            "preview": updatePreview,
        }

        def call_mappings(in_dict, map_slice):
            for key in in_dict:
                if key in map_slice:
                    if callable(map_slice[key]):
                        map_slice[key](in_dict[key])
                    elif isinstance(map_slice[key], dict):
                        call_mappings(in_dict[key], map_slice[key])

        call_mappings(test_data, mapping)
        if "preview" in test_data:
            del test_data["preview"]
        if "depth" in test_data:
            if "confidence" in test_data["depth"]:
                del test_data["depth"]["confidence"]
            if "subpixel" in test_data["depth"]:
                del test_data["depth"]["subpixel"]
            if "median" in test_data["depth"]:
                del test_data["depth"]["median"]
            if "sigma" in test_data["depth"]:
                del test_data["depth"]["sigma"]
        print(test_data)
        self.server.instance.restartDemo()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def stream(self):
        try:
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                if self.server.frametosend is not None:
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


class CustomHTTPServer(HTTPServer):
    instance = None
    config = {}
    frametosend = None

    def __init__(self, instance, handler):
        super().__init__((instance.confManager.args.host, instance.confManager.args.port), handler)
        self.instance = instance

    def finish_request(self, request, client_address):
        request.settimeout(1) # Really short timeout as there is only 1 thread
        try:
            HTTPServer.finish_request(self, request, client_address)
        except OSError:
            pass

class WebApp:
    def __init__(self, instance, args):
        super().__init__()
        self.confManager = prepareConfManager(args)
        self.running = False
        self.webserver = None
        self.selectedPreview = self.confManager.args.show[0] if len(self.confManager.args.show) > 0 else "color"
        self._demoInstance = instance
        self.thread = None

    def updateArg(self, arg_name, arg_value):
        setattr(self.confManager.args, arg_name, arg_value)

    def shouldRun(self):
        return self.running

    def onShowFrame(self, frame, source):
        if source == self.selectedPreview:
            self.webserver.frametosend = frame

    def onSetup(self, instance):
        try:
            with Path(projectRoot / ".consent").open() as f:
                statisticsEnabled = json.load(f)["statistics"]
        except:
            statisticsEnabled = True
        previewChoices = self.confManager.args.show
        devices = list(map(lambda info: info.getMxId(), dai.Device.getAllAvailableDevices()))
        countLabels = instance._nnManager._labels if instance._nnManager is not None else []
        countLabel = instance._nnManager._countLabel if instance._nnManager is not None else None
        depthEnabled = self.confManager.useDepth
        modelChoices = sorted(self.confManager.getAvailableZooModels(), key=cmp_to_key(lambda a, b: -1 if a == "mobilenet-ssd" else 1 if b == "mobilenet-ssd" else -1 if a < b else 1))
        medianChoices = list(filter(lambda name: name.startswith('KERNEL_') or name.startswith('MEDIAN_'), vars(dai.MedianFilter).keys()))

        self.webserver.config = {
            "ai": {
                "enabled": self.confManager.useNN,
                "model": {
                    "current": self.confManager.getModelName(),
                    "available": modelChoices,
                },
                "fullFov": not self.confManager.args.disableFullFovNn,
                "source": {
                    "current": self.confManager.getModelSource(),
                    "available": cameraChoices
                },
                "shaves": self.confManager.shaves,
                "ovVersion": {
                    "current": instance._pm.pipeline.getOpenVINOVersion().name.replace("VERSION_", ""),
                    "available": openvinoVersions,
                },
                "label": {
                    "current": countLabel,
                    "available": countLabels,
                },
                "sbb": self.confManager.args.spatialBoundingBox,
                "sbbFactor": self.confManager.args.sbbScaleFactor,
            },
            "depth": {
                "enabled": depthEnabled,
                "median": {
                    "current": self.confManager.args.stereoMedianSize,
                    "available": medianChoices
                },
                "subpixel": self.confManager.args.subpixel,
                "lrc": self.confManager.args.stereoMedianSize,
                "extended": self.confManager.args.extendedDisparity,
                "confidence": self.confManager.args.disparityConfidenceThreshold,
                "sigma": self.confManager.args.sigma,
                "lrcThreshold": self.confManager.args.lrcThreshold,
                "range": {
                    "min": self.confManager.args.minDepth,
                    "max": self.confManager.args.maxDepth,
                },
            },
            "camera": {
                "sync": self.confManager.args.sync,
                "color": {
                    "fps": self.confManager.args.rgbFps,
                    "resolution": self.confManager.args.rgbResolution,
                    "iso": self.confManager.args.cameraSensitivity,
                    "exposure": self.confManager.args.cameraExposure,
                    "saturation": self.confManager.args.cameraSaturation,
                    "contrast": self.confManager.args.cameraContrast,
                    "brightness": self.confManager.args.cameraBrightness,
                    "sharpness": self.confManager.args.cameraSharpness,
                },
                "mono": {
                    "fps": self.confManager.args.monoFps,
                    "resolution": self.confManager.args.monoResolution,
                    "iso": self.confManager.args.cameraSensitivity,
                    "exposure": self.confManager.args.cameraExposure,
                    "saturation": self.confManager.args.cameraSaturation,
                    "contrast": self.confManager.args.cameraContrast,
                    "brightness": self.confManager.args.cameraBrightness,
                    "sharpness": self.confManager.args.cameraSharpness,
                }
            },
            "misc": {
                "recording": {
                    "color": self.confManager.args.encode.get("color", None) if self.confManager.args.encode is not None else None,
                    "left": self.confManager.args.encode.get("left", None) if self.confManager.args.encode is not None else None,
                    "right": self.confManager.args.encode.get("right", None) if self.confManager.args.encode is not None else None,
                    "dest": str(self.confManager.args.encodeOutput),
                },
                "reporting": {
                    "enabled": self.confManager.args.report,
                    "dest": str(self.confManager.args.reportFile)
                },
                "demo": {
                    "statistics": statisticsEnabled,
                }
            },
            "preview": {
                "current": self.selectedPreview,
                "available": previewChoices,
            },
            "devices": {
                "current": instance._deviceInfo.getMxId(),
                "available": devices,
            },
            "depthEnabled": depthEnabled,
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
            self.webserver = CustomHTTPServer(self, HttpHandler)
            print("Server started http://{}:{}".format(self.confManager.args.host, self.confManager.args.port))

            try:
                self.webserver.serve_forever()
            except KeyboardInterrupt:
                pass

            self.webserver.server_close()
        else:
            self.webserver.frametosend = None
            self.webserver.config = {}

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

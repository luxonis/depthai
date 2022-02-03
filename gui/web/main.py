# This Python file uses the following encoding: utf-8
import asyncio
import json
import mimetypes
import sys
import threading
import time
import traceback
import urllib.parse
from functools import cmp_to_key
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path

import aiohttp
from aiohttp import web, MultipartWriter
from PIL import Image
import cv2
import depthai as dai
from depthai_sdk import createBlankFrame, Previews
from depthai_helpers.arg_manager import openvinoVersions, colorMaps, streamChoices, cameraChoices, reportingChoices, \
    projectRoot
from depthai_helpers.config_manager import prepareConfManager

def merge(source, destination):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


class HttpHandler:
    static_path = Path(__file__).parent / "dist"
    instance = None
    runner = None
    site = None
    loop = None
    config = {}
    frametosend = None
    app = None

    def __init__(self, instance, loop):
        self.instance = instance
        self.loop = loop
        self.app = web.Application(middlewares=[self.static_serve])
        self.app.add_routes([
            web.get('/stream', self.stream),
            web.get('/config', self.getConfig),
            web.post('/update', self.update),
        ])

    @web.middleware
    async def static_serve(self, request, handler):
        relative_file_path = Path(request.path).relative_to('/')  # remove root '/'
        file_path = self.static_path / relative_file_path  # rebase into static dir
        if not file_path.exists():
            return await handler(request)
        if file_path.is_dir():
            file_path /= 'index.html'
            if not file_path.exists():
                return web.HTTPNotFound()
        return web.FileResponse(file_path)

    def run(self):
        self.runner = web.AppRunner(self.app)
        self.loop.run_until_complete(self.runner.setup())
        self.site = aiohttp.web.TCPSite(self.runner, self.instance.confManager.args.host, self.instance.confManager.args.port)
        self.loop.run_until_complete(self.site.start())
        self.loop.run_forever()

    def close(self):
        self.loop.run_until_complete(self.runner.cleanup())

    async def getConfig(self, request):
        return web.json_response(self.config)

    async def update(self, request):
        data = await request.json()
        qs = request.query

        def updatePreview(data):
            self.instance.selectedPreview = data

        def updateStatistics(data):
            try:
                with Path(projectRoot / ".consent").open('w') as f:
                    json.dump({"statistics": data}, f)
            except:
                pass

        def updateCam(name, fps=None, resolution=None, exposure=None, iso=None, saturation=None, contrast=None, brightness=None, sharpness=None):
            if fps is not None:
                self.instance.updateArg("rgbFps" if name == Previews.color.name else "monoFps", int(fps))
            if resolution is not None and "current" in resolution:
                self.instance.updateArg("rgbResolution" if name == Previews.color.name else "monoResolution", resolution["current"])
            if exposure is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.instance.confManager.args.cameraExposure or []))) + [(name, int(exposure))]
                self.instance._demoInstance._cameraConfig["exposure"] = newValue
                self.instance.updateArg("cameraExposure", newValue)
            if iso is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.instance.confManager.args.cameraSensitivity or []))) + [(name, int(iso))]
                self.instance._demoInstance._cameraConfig["sensitivity"] = newValue
                self.instance.updateArg("cameraSensitivity", newValue)
            if saturation is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.instance.confManager.args.cameraSaturation or []))) + [(name, int(saturation))]
                self.instance._demoInstance._cameraConfig["saturation"] = newValue
                self.instance.updateArg("cameraSaturation", newValue)
            if contrast is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.instance.confManager.args.cameraContrast or []))) + [(name, int(contrast))]
                self.instance._demoInstance._cameraConfig["contrast"] = newValue
                self.instance.updateArg("cameraContrast", newValue, False)
            if brightness is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.instance.confManager.args.cameraBrightness or []))) + [(name, int(brightness))]
                self.instance._demoInstance._cameraConfig["brightness"] = newValue
                self.instance.updateArg("cameraBrightness", newValue, False)
            if sharpness is not None:
                newValue = list(filter(lambda item: item[0] == name, (self.instance.confManager.args.cameraSharpness or []))) + [(name, sharpness)]
                self.instance._demoInstance._cameraConfig["sharpness"] = newValue
                self.instance.updateArg("cameraSharpness", newValue, False)

            self.instance._demoInstance._updateCameraConfigs()

        mapping = {
            "ai": {
                "enabled": lambda data: self.instance.updateArg("disableNeuralNetwork", not data),
                "model": lambda data: self.instance.updateArg("cnnModel", data["current"]) if "current" in data else None,
                "fullFov": lambda data: self.instance.updateArg("disableFullFovNn", not data),
                "source": lambda data: self.instance.updateArg("camera", data["current"]),
                "shaves": lambda data: self.instance.updateArg("shaves", data),
                "ovVersion": lambda data: self.instance.updateArg("openvinoVersion", data["current"]) if "current" in data else None,
                "label": lambda data: self.instance.updateArg("countLabel", data["current"]) if "current" in data else None,
                "sbb": lambda data: self.instance.updateArg("spatialBoundingBox", data),
                "sbbFactor": lambda data: self.instance.updateArg("sbbScaleFactor", data),
            },
            "depth": {
                "enabled": lambda data: self.instance.updateArg("disableDepth", not data),
                "median": lambda data: self.instance.updateArg("stereoMedianSize", data["current"]) if "current" in data else None,
                "subpixel": lambda data: self.instance.updateArg("subpixel", data),
                "lrc": lambda data: self.instance.updateArg("disableStereoLrCheck", not data),
                "extended": lambda data: self.instance.updateArg("extendedDisparity", data),
                "confidence": lambda data: self.instance.updateArg("disparityConfidenceThreshold", data),
                "sigma": lambda data: self.instance.updateArg("sigma", data),
                "lrcThreshold": lambda data: self.instance.updateArg("lrcThreshold", data),
                "range": {
                    "min": lambda data: self.instance.updateArg("minDepth", data),
                    "max": lambda data: self.instance.updateArg("maxDepth", data),
                },
            },
            "camera": {
                "sync": lambda data: self.instance.updateArg("sync", data),
                "color": lambda data: updateCam(Previews.color.name, **data),
                "mono": lambda data: [updateCam(Previews.left.name, **data), updateCam(Previews.right.name, **data)]
            },
            "misc": {
                "recording": {
                    "color": lambda data: self.instance.updateArg("encode", {**self.instance.confManager.args.encode, "color": data}),
                    "left": lambda data: self.instance.updateArg("left", {**self.instance.confManager.args.encode, "left": data}),
                    "right": lambda data: self.instance.updateArg("right", {**self.instance.confManager.args.encode, "right": data}),
                    "dest": lambda data: self.instance.updateArg("encodeOutput", data),
                },
                "reporting": {
                    "enabled": lambda data: self.instance.updateArg("report", data),
                    "dest": lambda data: self.instance.updateArg("reportFile", data),
                },
                "demo": {
                    "statistics": updateStatistics,
                }
            },
            "preview": {
                "current": updatePreview,
            },
            "app": lambda data: self.instance.updateArg("app", data)
        }

        def call_mappings(in_dict, map_slice):
            for key in in_dict:
                if key in map_slice:
                    if callable(map_slice[key]):
                        map_slice[key](in_dict[key])
                    elif isinstance(map_slice[key], dict):
                        call_mappings(in_dict[key], map_slice[key])

        call_mappings(data, mapping)
        print(data)
        if "depth" in data:
            median = None
            if "median" in data["depth"] and "current" in data["depth"]["median"]:
                if data["depth"]["median"]["current"] == 3:
                    median = dai.MedianFilter.KERNEL_3x3
                elif data["depth"]["median"]["current"] == 5:
                    median = dai.MedianFilter.KERNEL_5x5
                elif data["depth"]["median"]["current"] == 7:
                    median = dai.MedianFilter.KERNEL_7x7
                else:
                    median = dai.MedianFilter.MEDIAN_OFF

            self.instance._demoInstance._pm.updateDepthConfig(
                self.instance._demoInstance._device, median=median, dct=data["depth"].get("confidence", None),
                sigma=data["depth"].get("sigma", None), lrcThreshold=data["depth"].get("lrcThreshold", None)
            )

        print(qs)
        if "restartRequired" in qs and qs["restartRequired"] == 'true':
            self.instance.restartDemo()
        else:
            self.config = merge(data, self.config)

        return web.Response()

    async def stream(self, request):
        boundary = 'boundarydonotcross'
        encode_param = (int(cv2.IMWRITE_JPEG_QUALITY), 90)
        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=--{}'.format(boundary),
        })
        try:
            await response.prepare(request)
            while True:
                if self.frametosend is not None:
                    with MultipartWriter('image/jpeg', boundary=boundary) as mpwriter:
                        result, encimg = cv2.imencode('.jpg', self.frametosend, encode_param)
                        data = encimg.tostring()
                        mpwriter.append(data, {
                            'Content-Type': 'image/jpeg'
                        })
                        await mpwriter.write(response, close_boundary=False)
                    await response.drain()
        except ConnectionResetError:
            print("Client connection closed")
        finally:
            return response


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

        defaultRecordingConfig = {"enabled": False, "fps": 30}

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
                    "iso": next(map(lambda data: data[1], filter(lambda data: data[0] == "color", self.confManager.args.cameraSensitivity or [])), None),
                    "exposure": next(map(lambda data: data[1], filter(lambda data: data[0] == "color", self.confManager.args.cameraExposure or [])), None),
                    "saturation": next(map(lambda data: data[1], filter(lambda data: data[0] == "color", self.confManager.args.cameraSaturation or [])), None),
                    "contrast": next(map(lambda data: data[1], filter(lambda data: data[0] == "color", self.confManager.args.cameraContrast or [])), None),
                    "brightness": next(map(lambda data: data[1], filter(lambda data: data[0] == "color", self.confManager.args.cameraBrightness or [])), None),
                    "sharpness": next(map(lambda data: data[1], filter(lambda data: data[0] == "color", self.confManager.args.cameraSharpness or [])), None),
                },
                "mono": {
                    "fps": self.confManager.args.monoFps,
                    "resolution": self.confManager.args.monoResolution,
                    "iso": next(map(lambda data: data[1], filter(lambda data: data[0] != "color", self.confManager.args.cameraSensitivity or [])), None),
                    "exposure": next(map(lambda data: data[1], filter(lambda data: data[0] != "color", self.confManager.args.cameraExposure or [])), None),
                    "saturation": next(map(lambda data: data[1], filter(lambda data: data[0] != "color", self.confManager.args.cameraSaturation or [])), None),
                    "contrast": next(map(lambda data: data[1], filter(lambda data: data[0] != "color", self.confManager.args.cameraContrast or [])), None),
                    "brightness": next(map(lambda data: data[1], filter(lambda data: data[0] != "color", self.confManager.args.cameraBrightness or [])), None),
                    "sharpness": next(map(lambda data: data[1], filter(lambda data: data[0] != "color", self.confManager.args.cameraSharpness or [])), None),
                }
            },
            "misc": {
                "recording": {
                    "color": {
                        "enabled": "color" in self.confManager.args.encode,
                        "fps": self.confManager.args.encode.get("color", 30),
                    } if self.confManager.args.encode is not None else defaultRecordingConfig,
                    "left": {
                        "enabled": "left" in self.confManager.args.encode,
                        "fps": self.confManager.args.encode.get("left", 30),
                    } if self.confManager.args.encode is not None else defaultRecordingConfig,
                    "right": {
                        "enabled": "right" in self.confManager.args.encode,
                        "fps": self.confManager.args.encode.get("right", 30),
                    } if self.confManager.args.encode is not None else defaultRecordingConfig,
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
            "app": None
        }


    def onAppSetup(self, app):
        self.webserver.config["app"] = {
            "name": app.appName,
            "state": "setup"
        }

    def onAppStart(self, app):
        self.webserver.config["app"] = {
            "name": app.appName,
            "state": "running"
        }

    def onError(self, ex: Exception):
        exception_message = ''.join(traceback.format_tb(ex.__traceback__) + [str(ex)])
        print(exception_message)

    def runDemo(self):
        self._demoInstance.setCallbacks(
            shouldRun=self.shouldRun, onShowFrame=self.onShowFrame, onSetup=self.onSetup, onAppSetup=self.onAppSetup,
            onAppStart=self.onAppStart
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
            loop = asyncio.get_event_loop()
            self.webserver = HttpHandler(self, loop)
            print("Server started http://{}:{}".format(self.confManager.args.host, self.confManager.args.port))

            try:
                self.webserver.run()
            except KeyboardInterrupt:
                pass

            self.webserver.close()
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

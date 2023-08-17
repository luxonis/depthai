import os
import platform
import subprocess
from pathlib import Path
import cv2
import depthai as dai
import numpy as np

from depthai_helpers.cli_utils import cliPrint, PrintColors
from depthai_sdk.previews import Previews


DEPTHAI_ZOO = Path(__file__).parent.parent / Path(f"resources/nn/")
DEPTHAI_VIDEOS = Path(__file__).parent.parent / Path(f"videos/")
DEPTHAI_VIDEOS.mkdir(exist_ok=True)


class ConfigManager:
    labels = ""
    customFwCommit = ''

    def __init__(self, args):
        self.args = args 

        # Get resolution width as it's required by some functions
        self.rgbResWidth = self.rgbResolutionWidth(self.args.rgbResolution)

        self.args.encode = dict(self.args.encode)
        self.args.cameraOrientation = dict(self.args.cameraOrientation)
        if (Previews.left.name in self.args.cameraOrientation or Previews.right.name in self.args.cameraOrientation) and self.useDepth:
            print("[WARNING] Changing mono cameras orientation may result in incorrect depth/disparity maps")

    def rgbResolutionWidth(self, res: dai.ColorCameraProperties.SensorResolution) -> int:
        if res == dai.ColorCameraProperties.SensorResolution.THE_720_P: return 720
        elif res == dai.ColorCameraProperties.SensorResolution.THE_800_P: return 800
        elif res == dai.ColorCameraProperties.SensorResolution.THE_1080_P: return 1080
        elif res == dai.ColorCameraProperties.SensorResolution.THE_4_K: return 2160
        elif res == dai.ColorCameraProperties.SensorResolution.THE_12_MP: return 3040
        elif res == dai.ColorCameraProperties.SensorResolution.THE_13_MP: return 3120
        else: raise Exception('Resolution not supported!')

    # Not needed, but might be useful for SDK in the future
    # def _monoResWidth(self, res: dai.MonoCameraProperties.SensorResolution) -> int:
    #     if res == dai.MonoCameraProperties.SensorResolution.THE_400_P: return 400
    #     elif res == dai.MonoCameraProperties.SensorResolution.THE_480_P: return 480
    #     elif res == dai.MonoCameraProperties.SensorResolution.THE_720_P: return 720
    #     elif res == dai.MonoCameraProperties.SensorResolution.THE_800_P: return 800
    #     else: raise Exception('Resolution not supported!')

    @property
    def debug(self):
        return not self.args.noDebug

    @property
    def useCamera(self):
        return not self.args.video

    @property
    def useNN(self):
        return not self.args.disableNeuralNetwork

    @property
    def useDepth(self):
        return not self.args.disableDepth and self.useCamera

    @property
    def maxDisparity(self):
        maxDisparity = 95
        if (self.args.extendedDisparity):
            maxDisparity *= 2
        if (self.args.subpixel):
            maxDisparity *= 32

        return maxDisparity

    def getModelSource(self):
        if not self.useCamera:
            return "host"
        if self.args.camera == "left":
            if self.useDepth:
                return "rectifiedLeft"
            return "left"
        if self.args.camera == "right":
            if self.useDepth:
                return "rectifiedRight"
            return "right"
        if self.args.camera == "color":
            return "color"

    def irEnabled(self, device):
        try:
            drivers = device.getIrDrivers()
            return len(drivers) > 0
        except RuntimeError:
            return False

    def getModelName(self):
        if self.args.cnnModel:
            return self.args.cnnModel
        modelDir = self.getModelDir()
        if modelDir is not None:
            return Path(modelDir).stem

    def getModelDir(self):
        if self.args.cnnPath:
            return self.args.cnnPath
        if self.args.cnnModel is not None and (DEPTHAI_ZOO / self.args.cnnModel).exists():
            return DEPTHAI_ZOO / self.args.cnnModel

    def getAvailableZooModels(self):
        def verify(path: Path):
            return path.parent.name == path.stem

        def convert(path: Path):
            return path.stem

        return list(map(convert, filter(verify, DEPTHAI_ZOO.rglob("**/*.json"))))

    def getColorMap(self):
        cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), getattr(cv2, "COLORMAP_{}".format(self.args.colorMap)))
        cvColorMap[0] = [0, 0, 0]
        return cvColorMap

    def getUsb2Mode(self):
        if self.args['forceUsb2']:
            cliPrint("FORCE USB2 MODE", PrintColors.WARNING)
            usb2Mode = True
        else:
            usb2Mode = False
        return usb2Mode

    def adjustPreviewToOptions(self):
        if len(self.args.show) != 0:
            depthPreviews = [Previews.rectifiedRight.name, Previews.rectifiedLeft.name, Previews.depth.name,
                             Previews.depthRaw.name, Previews.disparity.name, Previews.disparityColor.name]

            if len([preview for preview in self.args.show if preview in depthPreviews]) == 0 and not self.useNN:
                print("No depth-related previews chosen, disabling depth...")
                self.args.disableDepth = True
            return

        self.args.show.append(Previews.color.name)
        if self.useDepth:
            self.args.show.append(Previews.disparityColor.name)

        if self.args.guiType == "qt":
            if self.useNN:
                self.args.show.append(Previews.nnInput.name)

            if self.useDepth:
                if self.lowBandwidth:
                    self.args.show.append(Previews.disparityColor.name)
                else:
                    self.args.show.append(Previews.depthRaw.name)
                self.args.show.append(Previews.rectifiedLeft.name)
                self.args.show.append(Previews.rectifiedRight.name)
            else:
                self.args.show.append(Previews.left.name)
                self.args.show.append(Previews.right.name)

    def adjustParamsToDevice(self, device):
        deviceInfo = device.getDeviceInfo()
        cams = device.getConnectedCameras()
        depthEnabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams

        sensorNames = device.getCameraSensorNames()
        if dai.CameraBoardSocket.RGB in cams:
            name = sensorNames[dai.CameraBoardSocket.RGB]
            if name == 'OV9782':
                if self.rgbResWidth not in [720, 800]:
                    self.args.rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_800_P
                    cliPrint(f'{name} requires 720 or 800 resolution, defaulting to {self.args.rgbResolution}', 
                             PrintColors.RED)
            else:
                if self.rgbResWidth in [720, 800]:
                    self.args.rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
                    cliPrint(f'{name} doesn\'t support 720 / 800 resolutions, defaulting to {self.args.rgbResolution}', 
                             PrintColors.RED)

        if not depthEnabled:
            if not self.args.disableDepth:
                print("Disabling depth...")
                self.args.disableDepth = True
            if self.args.spatialBoundingBox:
                print("Disabling spatial bounding boxes...")
                self.args.spatialBoundingBox = False
            if self.args.camera != 'color':
                print("Switching source to RGB camera...")
                self.args.camera = 'color'
            updatedShowArg = []
            for name in self.args.show:
                if name in ("nnInput", "color"):
                    updatedShowArg.append(name)
                else:
                    print("Disabling {} preview...".format(name))
            if len(updatedShowArg) == 0:
                print("No previews available, adding defaults...")
                updatedShowArg.append("color")
                if self.useNN:
                    updatedShowArg.append("nnInput")
            self.args.show = updatedShowArg

        if self.args.bandwidth == "auto":
            if deviceInfo.protocol != dai.XLinkProtocol.X_LINK_USB_VSC:
                print("Enabling low-bandwidth mode due to connection mode... (protocol: {})".format(deviceInfo.protocol))
                self.args.bandwidth = "low"
                print("Setting PoE video quality to 50 to reduce latency...")
                self.args.poeQuality = 50
            elif device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
                print("Enabling low-bandwidth mode due to low USB speed... (speed: {})".format(device.getUsbSpeed()))
                self.args.bandwidth = "low"
            else:
                self.args.bandwidth = "high"

    def linuxCheckApplyUsbRules(self):
        if platform.system() == 'Linux':
            ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if(ret != 0):
                cliPrint("WARNING: Usb rules not found", PrintColors.WARNING)
                cliPrint("""
Run the following commands to set USB rules:

$ echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
$ sudo udevadm control --reload-rules && sudo udevadm trigger

After executing these commands, disconnect and reconnect USB cable to your OAK device""", PrintColors.RED)
                os._exit(1)

    def getCountLabel(self, nnetManager):
        if self.args.countLabel is None:
            return None

        if self.args.countLabel.isdigit():
            obj = nnetManager.getLabelText(int(self.args.countLabel)).lower()
            print(f"Counting number of {obj} in the frame")
            return obj
        else: return self.args.countLabel.lower()

    @property
    def leftCameraEnabled(self):
        return (self.args.camera == Previews.left.name and self.useNN) or \
               Previews.left.name in self.args.show or \
               Previews.rectifiedLeft.name in self.args.show or \
               self.useDepth

    @property
    def rightCameraEnabled(self):
        return (self.args.camera == Previews.right.name and self.useNN) or \
               Previews.right.name in self.args.show or \
               Previews.rectifiedRight.name in self.args.show or \
               self.useDepth

    @property
    def rgbCameraEnabled(self):
        return (self.args.camera == Previews.color.name and self.useNN) or \
               Previews.color.name in self.args.show

    @property
    def inputSize(self):
        return tuple(map(int, self.args.cnnInputSize.split('x'))) if self.args.cnnInputSize else None

    @property
    def previewSize(self):
        return (576, 320)

    @property
    def lowBandwidth(self):
        return self.args.bandwidth == "low"

    @property
    def lowCapabilities(self):
        return platform.machine().startswith("arm") or platform.machine().startswith("aarch")

    @property
    def shaves(self):
        if self.args.shaves is not None:
            return self.args.shaves
        if not self.useCamera:
            return 8
        if self.rgbResWidth > 1080:
            return 5
        return 6

    @property
    def dispMultiplier(self):
        val = 255 / self.maxDisparity
        return val



import os
import platform
import subprocess
from pathlib import Path
import cv2
import depthai as dai

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
        self.args.encode = dict(self.args.encode)
        self.args.cameraOrientation = dict(self.args.cameraOrientation)
        if self.args.scale is None:
            self.args.scale = {"color": 0.37 if not self.args.sync else 1}
        else:
            self.args.scale = dict(self.args.scale)
        if not self.useCamera and not self.args.sync:
            print("[WARNING] When using video file as an input, it's highly recommended to run the demo with \"--sync\" flag")
        if (Previews.left.name in self.args.cameraOrientation or Previews.right.name in self.args.cameraOrientation) and self.useDepth:
            print("[WARNING] Changing mono cameras orientation may result in incorrect depth/disparity maps")

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

    def getColorMap(self):
        return getattr(cv2, "COLORMAP_{}".format(self.args.colorMap))

    def getRgbResolution(self):
        if self.args.rgbResolution == 2160:
            return dai.ColorCameraProperties.SensorResolution.THE_4_K
        elif self.args.rgbResolution == 3040:
            return dai.ColorCameraProperties.SensorResolution.THE_12_MP
        else:
            return dai.ColorCameraProperties.SensorResolution.THE_1080_P

    def getMonoResolution(self):
        if self.args.monoResolution == 720:
            return dai.MonoCameraProperties.SensorResolution.THE_720_P
        elif self.args.monoResolution == 800:
            return dai.MonoCameraProperties.SensorResolution.THE_800_P
        else:
            return dai.MonoCameraProperties.SensorResolution.THE_400_P

    def getMedianFilter(self):
        if self.args.subpixel:
            return dai.MedianFilter.MEDIAN_OFF
        if self.args.stereoMedianSize == 3:
            return dai.MedianFilter.KERNEL_3x3
        elif self.args.stereoMedianSize == 5:
            return dai.MedianFilter.KERNEL_5x5
        elif self.args.stereoMedianSize == 7:
            return dai.MedianFilter.KERNEL_7x7
        else:
            return dai.MedianFilter.MEDIAN_OFF

    def getUsb2Mode(self):
        if self.args['forceUsb2']:
            cliPrint("FORCE USB2 MODE", PrintColors.WARNING)
            usb2Mode = True
        else:
            usb2Mode = False
        return usb2Mode

    def adjustPreviewToOptions(self):
        if len(self.args.show) != 0:
            return

        if self.args.camera == "color" and Previews.color.name not in self.args.show:
            self.args.show.append(Previews.color.name)
        if self.useDepth:
            if self.lowBandwidth and Previews.disparityColor.name not in self.args.show:
                self.args.show.append(Previews.disparityColor.name)
            elif not self.lowBandwidth and Previews.depth.name not in self.args.show:
                self.args.show.append(Previews.depth.name)
            if self.args.camera == "left" and Previews.rectifiedLeft.name not in self.args.show:
                self.args.show.append(Previews.rectifiedLeft.name)
            if self.args.camera == "right" and Previews.rectifiedRight.name not in self.args.show:
                self.args.show.append(Previews.rectifiedRight.name)
        else:
            if self.args.camera == "left" and Previews.left.name not in self.args.show:
                self.args.show.append(Previews.left.name)
            if self.args.camera == "right" and Previews.right.name not in self.args.show:
                self.args.show.append(Previews.right.name)

    def adjustParamsToDevice(self, device):
        deviceInfo = device.getDeviceInfo()
        cams = device.getConnectedCameras()
        depthEnabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams

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
                print("No previews available, adding color...")
                updatedShowArg.append("color")
            self.args.show = updatedShowArg

        if self.args.bandwidth == "auto":
            if deviceInfo.desc.protocol != dai.XLinkProtocol.X_LINK_USB_VSC:
                print("Enabling low-bandwidth mode due to connection mode... (protocol: {})".format(deviceInfo.desc.protocol))
                self.args.bandwidth = "low"
            elif device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
                print("Enabling low-bandwidth mode due to low USB speed... (speed: {})".format(device.getUsbSpeed()))
                self.args.bandwidth = "low"
            else:
                self.args.bandwidth = "high"


    def linuxCheckApplyUsbRules(self):
        if platform.system() == 'Linux':
            ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if(ret != 0):
                cliPrint("\nWARNING: Usb rules not found", PrintColors.WARNING)
                cliPrint("\nSet rules: \n"
                """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n"""
                "sudo udevadm control --reload-rules && sudo udevadm trigger \n"
                "Disconnect/connect usb cable on host! \n", PrintColors.RED)
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
        return self.inputSize or (576, 324)

    @property
    def lowBandwidth(self):
        return self.args.bandwidth == "low"

    @property
    def shaves(self):
        if self.args.shaves is not None:
            return self.args.shaves
        if not self.useCamera and not self.args.sync:
            return 8
        if self.args.rgbResolution > 1080:
            return 5
        return 6

    @property
    def dispMultiplier(self):
        val = 255 / self.maxDisparity
        return val



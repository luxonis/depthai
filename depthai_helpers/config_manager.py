import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path
import cv2
import depthai as dai

from depthai_helpers.cli_utils import cli_print, PrintColors
from depthai_helpers.managers import Previews


def show_progress(curr, max):
    done = int(50 * curr / max)
    sys.stdout.write("\r[{}{}] ".format('=' * done, ' ' * (50-done)) )
    sys.stdout.flush()


DEPTHAI_ZOO = Path(__file__).parent.parent / Path(f"resources/nn/")
DEPTHAI_VIDEOS = Path(__file__).parent.parent / Path(f"videos/")
DEPTHAI_VIDEOS.mkdir(exist_ok=True)


class ConfigManager:
    labels = ""
    custom_fw_commit = ''

    def __init__(self, args):
        self.args = args
        self.args.encode = dict(self.args.encode)
        self.args.scale = dict(self.args.scale)

    @property
    def debug(self):
        return not self.args.no_debug

    @property
    def useCamera(self):
        return not self.args.video

    @property
    def useNN(self):
        return not self.args.disable_neural_network

    @property
    def useDepth(self):
        return not self.args.disable_depth and self.useCamera

    @property
    def maxDisparity(self):
        max_disparity = 96
        if (self.args.extended_disparity):
            max_disparity *= 2
        if (self.args.subpixel):
            max_disparity *= 32

        return max_disparity

    def getModelSource(self):
        if not self.useCamera:
            return "host"
        if self.args.camera == "left":
            if self.useDepth:
                return "rectified_left"
            return "left"
        if self.args.camera == "right":
            if self.useDepth:
                return "rectified_right"
            return "right"
        if self.args.camera == "color":
            return "color"

    def getModelName(self):
        if self.args.cnn_model:
            return self.args.cnn_model
        model_dir = self.getModelDir()
        if model_dir is not None:
            return Path(model_dir).stem

    def getModelDir(self):
        if self.args.cnn_path:
            return self.args.cnn_path
        if self.args.cnn_model is not None and (DEPTHAI_ZOO / self.args.cnn_model).exists():
            return DEPTHAI_ZOO / self.args.cnn_model

    def getColorMap(self):
        return getattr(cv2, "COLORMAP_{}".format(self.args.color_map))

    def getRgbResolution(self):
        if self.args.rgb_resolution == 2160:
            return dai.ColorCameraProperties.SensorResolution.THE_4_K
        elif self.args.rgb_resolution == 3040:
            return dai.ColorCameraProperties.SensorResolution.THE_12_MP
        else:
            return dai.ColorCameraProperties.SensorResolution.THE_1080_P

    def getMonoResolution(self):
        if self.args.mono_resolution == 720:
            return dai.MonoCameraProperties.SensorResolution.THE_720_P
        elif self.args.mono_resolution == 800:
            return dai.MonoCameraProperties.SensorResolution.THE_800_P
        else:
            return dai.MonoCameraProperties.SensorResolution.THE_400_P

    def getMedianFilter(self):
        if self.args.subpixel:
            return dai.MedianFilter.MEDIAN_OFF
        if self.args.stereo_median_size == 3:
            return dai.MedianFilter.KERNEL_3x3
        elif self.args.stereo_median_size == 5:
            return dai.MedianFilter.KERNEL_5x5
        elif self.args.stereo_median_size == 7:
            return dai.MedianFilter.KERNEL_7x7
        else:
            return dai.MedianFilter.MEDIAN_OFF

    def getUsb2Mode(self):
        usb2_mode = False
        if self.args['force_usb2']:
            cli_print("FORCE USB2 MODE", PrintColors.WARNING)
            usb2_mode = True
        else:
            usb2_mode = False
        return usb2_mode

    def getCustomFirmwarePath(self, commit):
        fwdir = '.fw_cache/'
        if not os.path.exists(fwdir):
            os.mkdir(fwdir)
        fw_variant = ''
        if self.getUsb2Mode():
            fw_variant = 'usb2-'
        fname = 'depthai-' + fw_variant + commit + '.cmd'
        path = fwdir + fname
        if not Path(path).exists():
            url = 'https://artifacts.luxonis.com/artifactory/luxonis-myriad-snapshot-local/depthai-device-side/'
            url += commit + '/' + fname
            print('Downloading custom FW:', url)
            # Need this to avoid "HTTP Error 403: Forbidden"
            class CustomURLopener(urllib.request.FancyURLopener):
                version = "Mozilla/5.0"
                # FancyURLopener doesn't report by default errors like 404
                def http_error_default(self, url, fp, errcode, errmsg, headers):
                    raise ValueError(errcode)
            url_opener = CustomURLopener()
            with url_opener.open(url) as response, open(path, 'wb') as outf:
                outf.write(response.read())
        return path

    def getCommandFile(self):
        debug_mode = False
        cmd_file = ''
        if self.args['firmware'] != None:
            self.custom_fw_commit = self.args['firmware']
        if self.args['dev_debug'] == None:
            # Debug -debug flag NOT present, check first for custom firmware
            if self.custom_fw_commit == '':
                debug_mode = False
            else:
                debug_mode = True
                cmd_file = self.getCustomFirmwarePath(self.custom_fw_commit)
        elif self.args['dev_debug'] == '':
            # If just -debug flag is present -> cmd_file = '' (wait for device to be connected beforehand)
            debug_mode = True
        else:
            debug_mode = True
            cmd_file = self.args['dev_debug']

        return cmd_file, debug_mode

    def downloadYTVideo(self):
        def progress_func(stream, chunk, bytes_remaining):
            show_progress(stream.filesize - bytes_remaining, stream.filesize)

        try:
            from pytube import YouTube
        except ImportError as ex:
            raise RuntimeError("Unable to use YouTube video due to the following import error: {}".format(ex))
        path = None
        for _ in range(10):
            try:
                path = YouTube(self.args.video, on_progress_callback=progress_func).streams.first().download(output_path=DEPTHAI_VIDEOS)
            except urllib.error.HTTPError:
                # TODO remove when this issue is resolved - https://github.com/pytube/pytube/issues/990
                # Often, downloading YT video will fail with 404 exception, but sometimes it's successful
                pass
            else:
                break
        if path is None:
            raise RuntimeError("Unable to download YouTube video. Please try again")
        print("Youtube video downloaded.")
        self.args.video = path

    def adjustPreviewToOptions(self):
        if len(self.args.show) != 0:
            return

        if self.args.camera == "color" and "color" not in self.args.show:
            self.args.show.append("color")
        if self.args.camera == "left" and "left" not in self.args.show:
            self.args.show.append("left")
        if self.args.camera == "right" and "right" not in self.args.show:
            self.args.show.append("right")
        if self.useDepth:
            if self.lowBandwidth and "disparity_color" not in self.args.show:
                self.args.show.append("disparity_color")
            elif not self.lowBandwidth and "depth" not in self.args.show:
                self.args.show.append("depth")

    def adjustParamsToDevice(self, device):
        device_info = device.getDeviceInfo()
        cams = device.getConnectedCameras()
        depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams

        if not depth_enabled:
            if not self.args.disable_depth:
                print("Disabling depth...")
            self.args.disable_depth = True
            if self.args.spatial_bounding_box:
                print("Disabling spatial bounding boxes...")
            self.args.spatial_bounding_box = False
            if self.args.camera != 'color':
                print("Switching source to RGB camera...")
            self.args.camera = 'color'
            updated_show_arg = []
            for name in self.args.show:
                if name in ("nn_input", "color"):
                    updated_show_arg.append(name)
                else:
                    print("Disabling {} preview...".format(name))
            if len(updated_show_arg) == 0:
                print("No previews available, adding color...")
                updated_show_arg.append("color")
            self.args.show = updated_show_arg

        if device_info.desc.protocol != dai.XLinkProtocol.X_LINK_USB_VSC:
            print("Enabling low-bandwidth mode due to connection mode... (protocol: {})".format(device_info.desc.protocol))
            self.args.low_bandwidth = True
        elif device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
            print("Enabling low-bandwidth mode due to low USB speed... (speed: {})".format(device.getUsbSpeed()))
            self.args.low_bandwidth = True


    def linuxCheckApplyUsbRules(self):
        if platform.system() == 'Linux':
            ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if(ret != 0):
                cli_print("\nWARNING: Usb rules not found", PrintColors.WARNING)
                cli_print("\nSet rules: \n"
                """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n"""
                "sudo udevadm control --reload-rules && sudo udevadm trigger \n"
                "Disconnect/connect usb cable on host! \n", PrintColors.RED)
                os._exit(1)

    def getDeviceInfo(self):
        device_infos = dai.Device.getAllAvailableDevices()
        if len(device_infos) == 0:
            raise RuntimeError("No DepthAI device found!")
        else:
            print("Available devices:")
            for i, device_info in enumerate(device_infos):
                print(f"[{i}] {device_info.getMxId()} [{device_info.state.name}]")

            if self.args.device_id == "list":
                raise SystemExit(0)
            elif self.args.device_id is not None:
                matching_device = next(filter(lambda info: info.getMxId() == self.args.device_id, device_infos), None)
                if matching_device is None:
                    raise RuntimeError(f"No DepthAI device found with id matching {self.args.device_id} !")
                return matching_device
            elif len(device_infos) == 1:
                return device_infos[0]
            else:
                val = input("Which DepthAI Device you want to use: ")
                try:
                    return device_infos[int(val)]
                except:
                    raise ValueError("Incorrect value supplied: {}".format(val))

    def getCountLabel(self, nnet_manager):
        if self.args.count_label is None:
            return None

        if self.args.count_label.isdigit():
            obj = nnet_manager.get_label_text(int(self.args.count_label)).lower()
            print(f"Counting number of {obj} in the frame")
            return obj
        else: return self.args.count_label.lower()

    @property
    def leftCameraEnabled(self):
        return (self.args.camera == Previews.left.name and self.useNN) or \
               Previews.left.name in self.args.show or \
               Previews.rectified_left.name in self.args.show or \
               self.useDepth

    @property
    def rightCameraEnabled(self):
        return (self.args.camera == Previews.right.name and self.useNN) or \
               Previews.left.name in self.args.show or \
               Previews.rectified_left.name in self.args.show or \
               self.useDepth

    @property
    def rgbCameraEnabled(self):
        return (self.args.camera == Previews.color.name and self.useNN) or \
               Previews.color.name in self.args.show

    @property
    def inputSize(self):
        return tuple(map(int, self.args.cnn_input_size.split('x'))) if self.args.cnn_input_size else None

    @property
    def previewSize(self):
        return self.inputSize or (576, 324)
    @property
    def lowBandwidth(self):
        return self.args.low_bandwidth



import os
import platform
import subprocess
import sys
import urllib.request
from difflib import get_close_matches
from pathlib import Path

import cv2
import depthai as dai

import blobconverter

from depthai_helpers.cli_utils import cli_print, PrintColors

DEPTHAI_ZOO = Path(__file__).parent.parent / Path(f"resources/nn/")
DEPTHAI_VIDEOS = Path(__file__).parent.parent / Path(f"videos/")
DEPTHAI_VIDEOS.mkdir(exist_ok=True)


def show_progress(curr, max):
    done = int(50 * curr / max)
    sys.stdout.write("\r[{}{}] ".format('=' * done, ' ' * (50-done)) )
    sys.stdout.flush()


class ConfigManager:
    labels = ""
    custom_fw_commit = ''

    def __init__(self, args):
        self.args = args

    @property
    def debug(self):
        return not self.args.no_debug

    @property
    def useCamera(self):
        return not self.args.video

    @property
    def useHQ(self):
        return self.args.high_quality

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
        if self.args.stereo_median_size == 3:
            return dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
        elif self.args.stereo_median_size == 5:
            return dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
        elif self.args.stereo_median_size == 7:
            return dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
        else:
            return dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

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
        if self.args.camera == "color" and "color" not in self.args.show:
            self.args.show.append("color")
        if self.args.camera == "left" and "left" not in self.args.show:
            self.args.show.append("left")
        if self.args.camera == "right" and "right" not in self.args.show:
            self.args.show.append("right")
        if self.useDepth and "depth" not in self.args.show:
            self.args.show.append("depth")

    def adjustParamsToDevice(self, device):
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
            self.args.show = updated_show_arg

    def linuxCheckApplyUsbRules(self):
        if platform.system() == 'Linux':
            ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if(ret != 0):
                cli_print("\nWARNING: Usb rules not found", PrintColors.WARNING)
                cli_print("\nSet rules: \n"
                """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n"""
                "sudo udevadm control --reload-rules && udevadm trigger \n"
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


class BlobManager:
    def __init__(self, model_name=None, model_dir=None):
        self.model_dir = None
        self.zoo_dir = None
        self.config_file = None
        self.blob_path = None
        self.use_zoo = False
        self.use_blob = False
        self.zoo_models = [f.stem for f in DEPTHAI_ZOO.iterdir() if f.is_dir()]
        if model_dir is None:
            self.model_name = model_name
            self.use_zoo = True
        else:
            self.model_dir = Path(model_dir)
            self.zoo_dir = self.model_dir.parent
            self.model_name = model_name or self.model_dir.name
            self.config_file = self.model_dir / "model.yml"
            blob = next(self.model_dir.glob("*.blob"), None)
            if blob is not None:
                self.use_blob = True
                self.blob_path = blob
            if not self.config_file.exists():
                self.use_zoo = True


    def compile(self, shaves, openvino_version, target='auto'):
        version = openvino_version.name.replace("VERSION_", "").replace("_", ".")
        if self.use_blob:
            return self.blob_path
        elif self.use_zoo:
            try:
                self.blob_path = blobconverter.from_zoo(
                    name=self.model_name,
                    shaves=shaves,
                    version=version
                )
                return self.blob_path
            except Exception as e:
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if "not found in model zoo" in e.response.text:
                        all_models = set(self.zoo_models + blobconverter.zoo_list())
                        suggested = get_close_matches(self.model_name, all_models)
                        if len(suggested) > 0:
                            print("Model {} not found in model zoo. Did you mean: {} ?".format(self.model_name, " / ".join(suggested)), file=sys.stderr)
                        else:
                            print("Model {} not found in model zoo", file=sys.stderr)
                        raise SystemExit(1)
                    raise RuntimeError("Blob conversion failed with status {}! Error: \"{}\"".format(e.response.status_code, e.response.text))
                else:
                    raise
        else:
            self.blob_path = blobconverter.compile_blob(
                version=version,
                blob_name=self.model_name,
                req_data={
                    "name": self.model_name,
                    "use_zoo": True,
                },
                req_files={
                    'config': self.config_file,
                },
                data_type="FP16",
                shaves=shaves,
            )
            return self.blob_path

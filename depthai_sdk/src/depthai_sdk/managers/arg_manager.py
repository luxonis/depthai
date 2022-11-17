import argparse
import os
import sys
from pathlib import Path
import depthai as dai
from typing import Tuple

def getRgbResolution(resolution: str):
    """
    Parses Color camera resolution based on the string
    """
    resolution = str(resolution).upper()
    if resolution == '3120' or resolution == '13MP':
        return dai.ColorCameraProperties.SensorResolution.THE_13_MP
    elif resolution == '3040' or resolution == '12MP':
        return dai.ColorCameraProperties.SensorResolution.THE_12_MP
    elif resolution == '2160' or resolution == '4K':
        return dai.ColorCameraProperties.SensorResolution.THE_4_K
    # elif resolution == '1920' or resolution == '5MP':
    #     return dai.ColorCameraProperties.SensorResolution.THE_5_MP
    elif resolution == '800' or resolution == '800P':
        return dai.ColorCameraProperties.SensorResolution.THE_800_P
    elif resolution == '720' or resolution == '720P':
        return dai.ColorCameraProperties.SensorResolution.THE_720_P
    else: # Default
        return dai.ColorCameraProperties.SensorResolution.THE_1080_P

def getMonoResolution(resolution: str):
    """
    Parses Mono camera resolution based on the string
    """
    resolution = str(resolution).upper()
    if resolution == '800' or resolution == '800P':
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    elif resolution == '720' or resolution == '720P':
        return dai.MonoCameraProperties.SensorResolution.THE_720_P
    elif resolution == '480' or resolution == '480P':
        return dai.MonoCameraProperties.SensorResolution.THE_480_P
    else: # Default
        return dai.MonoCameraProperties.SensorResolution.THE_400_P

folderPath = Path(os.path.abspath(sys.argv[0])).parent
try:
    import cv2
    _colorMaps = list(map(lambda name: name[len("COLORMAP_"):], filter(lambda name: name.startswith("COLORMAP_"), vars(cv2))))
except:
    _colorMaps = None
_streamChoices = ("nnInput", "color", "left", "right", "depth", "depthRaw", "disparity", "disparityColor", "rectifiedLeft", "rectifiedRight")
_openvinoVersions = [v.replace("VERSION_", "") for v in vars(dai.OpenVINO.Version) if v.startswith("VERSION_")]
_orientationChoices = list(filter(lambda var: var[0].isupper(), vars(dai.CameraImageOrientation)))

def _checkRange(minVal, maxVal):
    def checkFn(value):
        ivalue = int(value)
        if minVal <= ivalue <= maxVal:
            return ivalue
        else:
            raise argparse.ArgumentTypeError(
                "{} is an invalid int value, must be in range {}..{}".format(value, minVal, maxVal)
            )

    return checkFn

def _commaSeparated(default, cast=str):
    def _fun(option):
        optionList = option.split(",")
        if len(optionList) not in [1, 2]:
            raise argparse.ArgumentTypeError(
                "{0} format is invalid. See --help".format(option)
            )
        elif len(optionList) == 1:
            return optionList[0], cast(default)
        else:
            try:
                return optionList[0], cast(optionList[1])
            except ValueError:
                raise argparse.ArgumentTypeError(
                    "In option: {0} {1} is not in a correct format!".format(option, optionList[1])
                )
    return _fun

def _orientationCast(arg):
    if not hasattr(dai.CameraImageOrientation, arg):
        raise argparse.ArgumentTypeError("Invalid camera orientation specified: '{}'. Available: {}".format(arg, _orientationChoices))
    return getattr(dai.CameraImageOrientation, arg)

def _checkEnum(enum):
    def _fun(value: str):
        try:
            return getattr(enum, value.upper())
        except:
            choices = [f"'{str(item).split('.')[-1]}'" for name, item in vars(enum).items() if name.isupper()]
            raise argparse.ArgumentTypeError(
                "{} option wasn't found in {} options! Choices: {}".format(value, enum, ', '.join(choices))
            )

    return _fun


class ArgsManager():
    @staticmethod
    def parseArgs(parser: argparse.ArgumentParser = None):
        """
        Creates Argument parser for common OAK device configuration

        Args:
            parser (argparse.ArgumentParser, optional): Use an existing parser. By default it creates a new ArgumentParser.
        """
        if parser is None:
            parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('-cam', '--camera', choices=["left", "right", "color"], default="color", help="Use one of DepthAI cameras for inference (conflicts with -vid)")
        parser.add_argument('-vid', '--video', type=str, help="Path to video file (or YouTube link) to be used for inference (conflicts with -cam)")
        parser.add_argument('-dd', '--disableDepth', action="store_true", help="Disable depth information")
        parser.add_argument('-dnn', '--disableNeuralNetwork', action="store_true", help="Disable neural network inference")
        parser.add_argument('-cnnp', '--cnnPath', type=Path, help="Path to cnn model directory to be run")
        parser.add_argument("-cnn", "--cnnModel", default="mobilenet-ssd", type=str,
                            help="Cnn model to run on DepthAI")
        parser.add_argument('-sh', '--shaves', type=int, help="Number of MyriadX SHAVEs to use for neural network blob compilation")
        parser.add_argument('-cnnsize', '--cnnInputSize', default=None, type=str,
                            help="Neural network input dimensions, in \"WxH\" format, e.g. \"544x320\"")
        
        # Color/Mono cam related arguments
        parser.add_argument("-rgbr", "--rgbResolution", default='1080P', type=str,
                            help="RGB cam res height: (1920x)1080, (3840x)2160, (4056x)3040, (1280x)720, (1280x)800. Default: %(default)s")
        parser.add_argument("-rgbf", "--rgbFps", default=30.0, type=float,
                            help="RGB cam fps: max 118.0 for H:1080, max 42.0 for H:2160. Default: %(default)s")
        parser.add_argument("-monor", "--monoResolution", default='400P', type=str,
                            help="Mono cam res height: (1280x)720, (1280x)800 or (640x)400. Default: %(default)s")
        parser.add_argument("-monof", "--monoFps", default=30.0, type=float,
                            help="Mono cam fps: max 60.0 for H:720 or H:800, max 120.0 for H:400. Default: %(default)s")
        parser.add_argument('-fps', '--fps', type=float, help='Camera FPS applied to all sensors')

        # ColorCamera ISP values
        parser.add_argument('-isp', '--ispScale', type=_commaSeparated(None), help="Sets ColorCamera's ISP scale")
        parser.add_argument('-sharpness', '--sharpness', default=None, type=_checkRange(0,4),
                            help="Sets ColorCamera's sharpness")
        parser.add_argument('-lumaDenoise', '--lumaDenoise', default=None, type=_checkRange(0, 4),
                            help="Sets ColorCamera's Luma denoise")
        parser.add_argument('-chromaDenoise', '--chromaDenoise', default=None, type=_checkRange(0, 4),
                            help="Sets ColorCamera's Chroma denoise")

        # ColorCamera controls
        parser.add_argument('-manualFocus', '--manualFocus', default=None, type=_checkRange(0, 255),
                            help="Specify a Lens Position between 0 and 255 to use manual focus. Otherwise, auto-focus is used by default.")
        parser.add_argument('-afMode', '--afMode', default=None, type=_checkEnum(dai.CameraControl.AutoFocusMode),
                            help="Specify the Auto Focus mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-awbMode', '--awbMode', default=None, type=_checkEnum(dai.CameraControl.AutoWhiteBalanceMode),
                            help="Specify the Auto White Balance mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-sceneMode', '--sceneMode', default=None, type=_checkEnum(dai.CameraControl.SceneMode),
                            help="Specify the Scene mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-abMode', '-antiBandingMode', '--antiBandingMode', default=None, type=_checkEnum(dai.CameraControl.AntiBandingMode),
                            help="Specify the Anti-Banding mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-effectMode', '--effectMode', default=None, type=_checkEnum(dai.CameraControl.EffectMode),
                            help="Specify the Effect mode for the ColorCamera. AUTO by default.")

        parser.add_argument("--cameraControls", action="store_true", help="Show camera configuration options in GUI and control them using keyboard")
        parser.add_argument("--cameraExposure", type=_commaSeparated("all", int), nargs="+", help="Specify camera saturation")
        parser.add_argument("--cameraSensitivity", type=_commaSeparated("all", int), nargs="+", help="Specify camera sensitivity")
        parser.add_argument("--cameraSaturation", type=_commaSeparated("all", int), nargs="+", help="Specify image saturation")
        parser.add_argument("--cameraContrast", type=_commaSeparated("all", int), nargs="+", help="Specify image contrast")
        parser.add_argument("--cameraBrightness", type=_commaSeparated("all", int), nargs="+", help="Specify image brightness")
        parser.add_argument("--cameraSharpness", type=_commaSeparated("all", int), nargs="+", help="Specify image sharpness")

        
        # Depth related arguments
        parser.add_argument("-dct", "--disparityConfidenceThreshold", default=245, type=_checkRange(0, 255),
                            help="Disparity confidence threshold, used for depth measurement. Default: %(default)s")
        parser.add_argument("-lrct", "--lrcThreshold", default=4, type=_checkRange(0, 10),
                            help="Left right check threshold, used for depth measurement. Default: %(default)s")
        parser.add_argument("-sig", "--sigma", default=0, type=_checkRange(0, 250),
                            help="Sigma value for Bilateral Filter applied on depth. Default: %(default)s")
        parser.add_argument("-med", "--stereoMedianSize", default=5, type=int, choices=[0, 3, 5, 7],
                            help="Disparity / depth median filter kernel size (N x N) . 0 = filtering disabled. Default: %(default)s")
        parser.add_argument('-dlrc', '--disableStereoLrCheck', action="store_false", dest="stereoLrCheck",
                            help="Disable stereo 'Left-Right check' feature.")
        parser.add_argument('-ext', '--extendedDisparity', action="store_true",
                            help="Enable stereo 'Extended Disparity' feature.")
        parser.add_argument('-sub', '--subpixel', action="store_true",
                            help="Enable stereo 'Subpixel' feature.")
        parser.add_argument("-cm", "--colorMap", default="JET", choices=_colorMaps, help="Change color map used to apply colors to depth/disparity frames. Default: %(default)s")
        
        # Spatial image detection related arguments
        parser.add_argument("-maxd", "--maxDepth", default=10000, type=int,
                            help="Maximum depth distance for spatial coordinates in mm. Default: %(default)s")
        parser.add_argument("-mind", "--minDepth", default=100, type=int,
                            help="Minimum depth distance for spatial coordinates in mm. Default: %(default)s")
        parser.add_argument('-sbb', '--spatialBoundingBox', action="store_true",
                            help="Display spatial bounding box (ROI) when displaying spatial information. The Z coordinate get's calculated from the ROI (average)")
        parser.add_argument("-sbbsf", "--sbbScaleFactor", default=0.3, type=float,
                            help="Spatial bounding box scale factor. Sometimes lower scale factor can give better depth (Z) result. Default: %(default)s")
        
        parser.add_argument('-s', '--show', default=[], nargs="+", choices=_streamChoices, help="Choose which previews to show. Default: %(default)s")
        parser.add_argument("-dff", "--disableFullFovNn", default=False, action="store_true",
                            help="Disable full RGB FOV for NN, keeping the nn aspect ratio")
        parser.add_argument('--report', nargs="+", default=[], choices=["temp", "cpu", "memory"], help="Display device utilization data")
        parser.add_argument('--reportFile', help="Save report data to specified target file in CSV format")
        parser.add_argument('-cb', '--callback', type=Path, default=folderPath / 'callbacks.py', help="Path to callbacks file to be used. Default: %(default)s")
        parser.add_argument("--openvinoVersion", type=str, choices=_openvinoVersions, help="Specify which OpenVINO version to use in the pipeline")
        parser.add_argument("--count", type=str, dest='countLabel',
                            help="Count and display the number of specified objects on the frame. You can enter either the name of the object or its label id (number).")
        parser.add_argument("-dev", "--deviceId", type=str,
                            help="DepthAI MX id of the device to connect to. Use the word 'list' to show all devices and exit.")
        parser.add_argument('-bandw', '--bandwidth', type=str, default="auto", choices=["auto", "low", "high"], help="Force bandwidth mode. \n"
                                                                                                                    "If set to \"high\", the output streams will stay uncompressed\n"
                                                                                                                    "If set to \"low\", the output streams will be MJPEG-encoded\n"
                                                                                                                    "If set to \"auto\" (default), the optimal bandwidth will be selected based on your connection type and speed")
        parser.add_argument('-gt', '--guiType', type=str, default="auto", choices=["auto", "qt", "cv"], help="Specify GUI type of the demo. \"cv\" uses built-in OpenCV display methods, \"qt\" uses Qt to display interactive GUI. \"auto\" will use OpenCV for Raspberry Pi and Qt for other platforms")
        parser.add_argument('-usbs', '--usbSpeed', type=str, default="usb3", choices=["usb2", "usb3"], help="Force USB communication speed. Default: %(default)s")
        parser.add_argument('-enc', '--encode', type=_commaSeparated(default=30.0, cast=float), nargs="+", default=[],
                            help="Define which cameras to encode (record) \n"
                                "Format: cameraName or cameraName,encFps \n"
                                "Example: -enc left color \n"
                                "Example: -enc color right,10 left,10")
        parser.add_argument('-encout', '--encodeOutput', type=Path, default=folderPath / 'recordings', help="Path to directory where to store encoded files. Default: %(default)s")
        parser.add_argument('-xls', '--xlinkChunkSize', type=int, help="Specify XLink chunk size")
        parser.add_argument('-poeq', '--poeQuality', type=_checkRange(1, 100), default=100, help="Specify PoE encoding video quality (1-100)")
        parser.add_argument('-camo', '--cameraOrientation', type=_commaSeparated(default="AUTO", cast=_orientationCast), nargs="+", default=[],
                            help=("Define cameras orientation (available: {}) \n"
                                "Format: camera_name,camera_orientation \n"
                                "Example: -camo color,ROTATE_180_DEG right,ROTATE_180_DEG left,ROTATE_180_DEG").format(', '.join(_orientationChoices))
                            )
        parser.add_argument("--irDotBrightness", type=_checkRange(0, 1200), default=0, help="For OAK-D Pro: specify IR dot projector brightness, range: 0..1200 [mA], default 0 (turned off)")
        parser.add_argument("--irFloodBrightness", type=_checkRange(0, 1500), default=0, help="For OAK-D Pro: specify IR flood illumination brightness, range: 0..1500 [mA], default 0 (turned off)")
        parser.add_argument('--skipVersionCheck', action="store_true", help="Disable libraries version check")
        parser.add_argument('--noSupervisor', action="store_true", help="Disable supervisor check")
        parser.add_argument('--sync', action="store_true", help="Enable frame and NN synchronization. If enabled, all frames and NN results will be synced before preview (same sequence number)")
        parser.add_argument('--noRgbDepthAlign', action="store_true", help="Disable RGB-Depth align (depth frame will be aligned with the RGB frame)")
        parser.add_argument('--debug', action="store_true", help="Enables debug mode. Capability to connect to already BOOTED devices and also implicitly disables version check")
        parser.add_argument("-app","--app", type=str, choices=["uvc", "record"], help="Specify which app to run instead of the demo")
        parser.add_argument('-tun', '--cameraTuning', type=Path, help="Path to camera tuning blob to use, overriding the built-in tuning")
        
        args = parser.parse_args()
        # Parse arguments
        args.rgbResolution = getRgbResolution(args.rgbResolution)
        args.monoResolution = getMonoResolution(args.monoResolution)
        # Global FPS setting, applied to all cameras
        if args.fps is not None:
            args.rgbFps = args.fps
            args.monoFps = args.fps
        return args

    @staticmethod
    def parseApp() -> str:
        """
        Returns app name specified in the arguments, or None, if no app was specified.
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-app","--app", type=str, choices=["uvc", "record"], help="Specify which app to run instead of the demo")
        known, unknown = parser.parse_known_args()
        return known.app



    

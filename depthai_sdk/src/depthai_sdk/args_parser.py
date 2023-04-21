import argparse
from pathlib import Path
from typing import Any, Dict

import depthai as dai

from depthai_sdk.components.parser import rgb_resolution, mono_resolution, parse_bool

"""
Very similar to ArgsManager, but specific to the new SDK, not depthai_demo.py. 
"""

# TODO: support changing colorMaps
# try:
#     import cv2
#     _colorMaps = list(
#         map(lambda name: name[len("COLORMAP_"):], filter(lambda name: name.startswith("COLORMAP_"), vars(cv2))))
# except:
#     _colorMaps = None

_openvinoVersions = [v.replace("VERSION_", "") for v in vars(dai.OpenVINO.Version) if v.startswith("VERSION_")]


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


class ArgsParser:
    @staticmethod
    def parseArgs(parser: argparse.ArgumentParser = None) -> Dict[str, Any]:
        """
        Creates Argument parser for common OAK device configuration

        Args:
            parser (argparse.ArgumentParser, optional): Use an existing parser. By default it creates a new ArgumentParser.
        """
        if parser is None:
            parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('-recording', '--recording', type=str,
                            help="Path or url to a depthai-recording/folder/YouTube url/file to be used instead of live camera feed")

        # Color/Mono cam related arguments
        parser.add_argument("-rgbr", "--rgbResolution", type=str,
                            help="RGB cam res height: (1920x)1080, (3840x)2160, (4056x)3040, (1280x)720, (1280x)800. Default: %(default)s")
        parser.add_argument("-rgbf", "--rgbFps", type=float,
                            help="RGB cam fps: max 118.0 for H:1080, max 42.0 for H:2160. Default: %(default)s")
        parser.add_argument("-monor", "--monoResolution", type=str,
                            help="Mono cam res height: (1280x)720, (1280x)800 or (640x)400. Default: %(default)s")
        parser.add_argument("-monof", "--monoFps", type=float,
                            help="Mono cam fps: max 60.0 for H:720 or H:800, max 120.0 for H:400. Default: %(default)s")
        parser.add_argument('-fps', '--fps', type=float, help='Camera FPS applied to all sensors')

        # ColorCamera ISP values
        parser.add_argument('-isp', '--ispScale', type=_commaSeparated(None), help="Sets ColorCamera's ISP scale")
        parser.add_argument('-sharpness', '--sharpness', type=_checkRange(0, 4),
                            help="Sets ColorCamera's sharpness")
        parser.add_argument('-lumaDenoise', '--lumaDenoise', type=_checkRange(0, 4),
                            help="Sets ColorCamera's Luma denoise")
        parser.add_argument('-chromaDenoise', '--chromaDenoise', type=_checkRange(0, 4),
                            help="Sets ColorCamera's Chroma denoise")

        # ColorCamera controls
        parser.add_argument('-manualFocus', '--manualFocus', type=_checkRange(0, 255),
                            help="Specify a Lens Position between 0 and 255 to use manual focus. Otherwise, auto-focus is used by default.")
        parser.add_argument('-afMode', '--afMode', type=_checkEnum(dai.CameraControl.AutoFocusMode),
                            help="Specify the Auto Focus mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-awbMode', '--awbMode',
                            type=_checkEnum(dai.CameraControl.AutoWhiteBalanceMode),
                            help="Specify the Auto White Balance mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-sceneMode', '--sceneMode', type=_checkEnum(dai.CameraControl.SceneMode),
                            help="Specify the Scene mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-abMode', '-antiBandingMode', '--antiBandingMode',
                            type=_checkEnum(dai.CameraControl.AntiBandingMode),
                            help="Specify the Anti-Banding mode for the ColorCamera. AUTO by default.")
        parser.add_argument('-effectMode', '--effectMode', type=_checkEnum(dai.CameraControl.EffectMode),
                            help="Specify the Effect mode for the ColorCamera. AUTO by default.")

        # parser.add_argument("--cameraControls",type=parse_bool,
        #                     help="Show camera configuration options in GUI and control them using keyboard")
        # parser.add_argument("--cameraExposure", type=_commaSeparated("all", int), nargs="+",
        #                     help="Specify camera saturation")
        # parser.add_argument("--cameraSensitivity", type=_commaSeparated("all", int), nargs="+",
        #                     help="Specify camera sensitivity")
        # parser.add_argument("--cameraSaturation", type=_commaSeparated("all", int), nargs="+",
        #                     help="Specify image saturation")
        # parser.add_argument("--cameraContrast", type=_commaSeparated("all", int), nargs="+",
        #                     help="Specify image contrast")
        # parser.add_argument("--cameraBrightness", type=_commaSeparated("all", int), nargs="+",
        #                     help="Specify image brightness")
        # parser.add_argument("--cameraSharpness", type=_commaSeparated("all", int), nargs="+",
        #                     help="Specify image sharpness")

        # StereoDepth related arguments
        parser.add_argument("-dct", "--disparityConfidenceThreshold", type=_checkRange(0, 255),
                            help="Disparity confidence threshold, used for depth measurement.")
        parser.add_argument("-lrct", "--lrcThreshold", type=_checkRange(0, 10),
                            help="Left right check threshold, used for depth measurement. Default: %(default)s")
        parser.add_argument("-sig", "--sigma", type=_checkRange(0, 250),
                            help="Sigma value for Bilateral Filter applied on depth. Default: %(default)s")
        parser.add_argument("-med", "--stereoMedianSize", type=int, choices=[0, 3, 5, 7],
                            help="Disparity / depth median filter kernel size (N x N) . 0 = filtering disabled. Default: %(default)s")
        parser.add_argument('-lrc', '--stereoLrCheck', type=parse_bool,
                            help="Set stereo LR-check feature.")
        parser.add_argument('-ext', '--extendedDisparity', type=parse_bool,
                            help="Set stereo Extended Disparity feature.")
        parser.add_argument('-sub', '--subpixel', type=parse_bool,
                            help="Set stereo Subpixel feature.")

        # parser.add_argument("-cm", "--colorMap",ET", choices=_colorMaps,
        #                     help="Change color map used to apply colors to depth/disparity frames. Default: %(default)s")

        # Spatial image detection related arguments
        parser.add_argument("-maxd", "--maxDepth", type=int,
                            help="Maximum depth distance for spatial coordinates in mm. Default: %(default)s")
        parser.add_argument("-mind", "--minDepth", type=int,
                            help="Minimum depth distance for spatial coordinates in mm. Default: %(default)s")
        # parser.add_argument('-sbb', '--spatialBoundingBox', type=parse_bool,
        #                     help="Display spatial bounding box (ROI) when displaying spatial information. The Z coordinate get's calculated from the ROI (average)")
        parser.add_argument("-sbbsf", "--sbbScaleFactor", type=float,
                            help="Spatial bounding box scale factor. Sometimes lower scale factor can give better depth (Z) result. Default: %(default)s")

        # Pipeline
        parser.add_argument("--openvinoVersion", type=str, choices=_openvinoVersions,
                            help="Specify which OpenVINO version to use in the pipeline")
        parser.add_argument('-xls', '--xlinkChunkSize', type=int, help="Specify XLink chunk size")
        parser.add_argument('-tun', '--cameraTuning', type=Path,
                            help="Path to camera tuning blob to use, overriding the built-in tuning")

        # Device
        parser.add_argument("-dev", "--deviceId", type=str,
                            help="DepthAI MX id of the device to connect to. Use the word 'list' to show all devices and exit.")
        parser.add_argument('-usbs', '--usbSpeed', type=str, choices=["usb2", "usb3"],
                            help="Force USB communication speed. Default: %(default)s")
        # Device after booting
        parser.add_argument("--irDotBrightness", "-laser", "--laser", type=_checkRange(0, 1200),
                            help="For OAK-D Pro: specify IR dot projector brightness, range: 0..1200 [mA], default 0 (turned off)")
        parser.add_argument("--irFloodBrightness", "-led", "--led", type=_checkRange(0, 1500),
                            help="For OAK-D Pro: specify IR flood illumination brightness, range: 0..1500 [mA], default 0 (turned off)")

        args = parser.parse_known_args()[0]
        # Parse arguments
        args.rgbResolution = rgb_resolution(args.rgbResolution)
        args.monoResolution = mono_resolution(args.monoResolution)
        # Global FPS setting, applied to all cameras
        if args.fps is not None:
            args.rgbFps = args.fps
            args.monoFps = args.fps

        args = vars(args)  # Namespace->Dict
        # Print user-defined arguments
        for name, val in args.items():
            if val is not None:
                print(f'{name}: {val}')
        return args

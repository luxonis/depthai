import os
import argparse
from pathlib import Path
import cv2
import depthai as dai
try:
    import argcomplete
except ImportError:
    raise ImportError('\033[1;5;31m argcomplete module not found, run: python3 install_requirements.py \033[0m')
from depthai_sdk.previews import Previews


def checkRange(minVal, maxVal):
    def checkFn(value):
        ivalue = int(value)
        if minVal <= ivalue <= maxVal:
            return ivalue
        else:
            raise argparse.ArgumentTypeError(
                "{} is an invalid int value, must be in range {}..{}".format(value, minVal, maxVal)
            )

    return checkFn


def _comaSeparated(default, cast=str):
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


orientationChoices = list(filter(lambda var: var[0].isupper(), vars(dai.CameraImageOrientation)))


def orientationCast(arg):
    if not hasattr(dai.CameraImageOrientation, arg):
        raise argparse.ArgumentTypeError("Invalid camera orientation specified: '{}'. Available: {}".format(arg, orientationChoices))

    return getattr(dai.CameraImageOrientation, arg)


openvinoVersions = list(map(lambda name: name.replace("VERSION_", ""), filter(lambda name: name.startswith("VERSION_"), vars(dai.OpenVINO.Version))))
_streamChoices = ("nnInput", "color", "left", "right", "depth", "depthRaw", "disparity", "disparityColor", "rectifiedLeft", "rectifiedRight")
colorMaps = list(map(lambda name: name[len("COLORMAP_"):], filter(lambda name: name.startswith("COLORMAP_"), vars(cv2))))
projectRoot = Path(__file__).parent.parent

def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-cam', '--camera', choices=[Previews.left.name, Previews.right.name, Previews.color.name], default=Previews.color.name, help="Use one of DepthAI cameras for inference (conflicts with -vid)")
    parser.add_argument('-vid', '--video', type=str, help="Path to video file (or YouTube link) to be used for inference (conflicts with -cam)")
    parser.add_argument('-dd', '--disableDepth', action="store_true", help="Disable depth information")
    parser.add_argument('-dnn', '--disableNeuralNetwork', action="store_true", help="Disable neural network inference")
    parser.add_argument('-cnnp', '--cnnPath', type=Path, help="Path to cnn model directory to be run")
    parser.add_argument("-cnn", "--cnnModel", default="mobilenet-ssd", type=str,
                        help="Cnn model to run on DepthAI")
    parser.add_argument('-sh', '--shaves', type=int, help="Number of MyriadX SHAVEs to use for neural network blob compilation")
    parser.add_argument('-cnnsize', '--cnnInputSize', default=None, type=str,
                        help="Neural network input dimensions, in \"WxH\" format, e.g. \"544x320\"")
    parser.add_argument("-rgbr", "--rgbResolution", default=1080, type=int, choices=[1080, 2160, 3040],
                        help="RGB cam res height: (1920x)1080, (3840x)2160 or (4056x)3040. Default: %(default)s")
    parser.add_argument("-rgbf", "--rgbFps", default=30.0, type=float,
                        help="RGB cam fps: max 118.0 for H:1080, max 42.0 for H:2160. Default: %(default)s")
    parser.add_argument("-dct", "--disparityConfidenceThreshold", default=245, type=checkRange(0, 255),
                        help="Disparity confidence threshold, used for depth measurement. Default: %(default)s")
    parser.add_argument("-lrct", "--lrcThreshold", default=4, type=checkRange(0, 10),
                        help="Left right check threshold, used for depth measurement. Default: %(default)s")
    parser.add_argument("-sig", "--sigma", default=0, type=checkRange(0, 250),
                        help="Sigma value for Bilateral Filter applied on depth. Default: %(default)s")
    parser.add_argument("-med", "--stereoMedianSize", default=7, type=int, choices=[0, 3, 5, 7],
                        help="Disparity / depth median filter kernel size (N x N) . 0 = filtering disabled. Default: %(default)s")
    parser.add_argument('-lrc', '--stereoLrCheck', action="store_true",
                        help="Enable stereo 'Left-Right check' feature.")
    parser.add_argument('-ext', '--extendedDisparity', action="store_true",
                        help="Enable stereo 'Extended Disparity' feature.")
    parser.add_argument('-sub', '--subpixel', action="store_true",
                        help="Enable stereo 'Subpixel' feature.")
    parser.add_argument("-dff", "--disableFullFovNn", default=False, action="store_true",
                        help="Disable full RGB FOV for NN, keeping the nn aspect ratio")
    parser.add_argument('-scale', '--scale', type=_comaSeparated(default=0.5, cast=float), nargs="+",
                        help="Define which preview windows to scale (grow/shrink). If scale_factor is not provided, it will default to 0.5 \n"
                             "Format: preview_name or preview_name,scale_factor \n"
                             "Example: -scale color \n"
                             "Example: -scale color,0.7 right,2 left,2")
    parser.add_argument("-cm", "--colorMap", default="JET", choices=colorMaps, help="Change color map used to apply colors to depth/disparity frames. Default: %(default)s")
    parser.add_argument("-maxd", "--maxDepth", default=10000, type=int,
                        help="Maximum depth distance for spatial coordinates in mm. Default: %(default)s")
    parser.add_argument("-mind", "--minDepth", default=100, type=int,
                        help="Minimum depth distance for spatial coordinates in mm. Default: %(default)s")
    parser.add_argument('-sbb', '--spatialBoundingBox', action="store_true",
                        help="Display spatial bounding box (ROI) when displaying spatial information. The Z coordinate get's calculated from the ROI (average)")
    parser.add_argument("-sbbsf", "--sbbScaleFactor", default=0.3, type=float,
                        help="Spatial bounding box scale factor. Sometimes lower scale factor can give better depth (Z) result. Default: %(default)s")
    parser.add_argument('-s', '--show', default=[], nargs="+", choices=_streamChoices, help="Choose which previews to show. Default: %(default)s")
    parser.add_argument('--report', nargs="+", default=[], choices=["temp", "cpu", "memory"], help="Display device utilization data")
    parser.add_argument('--reportFile', help="Save report data to specified target file in CSV format")
    parser.add_argument('-sync', '--sync', action="store_true",
                        help="Enable NN/camera synchronization. If enabled, camera source will be from the NN's passthrough attribute")
    parser.add_argument("-monor", "--monoResolution", default=400, type=int, choices=[400,720,800],
                        help="Mono cam res height: (1280x)720, (1280x)800 or (640x)400. Default: %(default)s")
    parser.add_argument("-monof", "--monoFps", default=30.0, type=float,
                        help="Mono cam fps: max 60.0 for H:720 or H:800, max 120.0 for H:400. Default: %(default)s")
    parser.add_argument('-cb', '--callback', type=Path, default=projectRoot / 'callbacks.py', help="Path to callbacks file to be used. Default: %(default)s")
    parser.add_argument("--openvinoVersion", type=str, choices=openvinoVersions, help="Specify which OpenVINO version to use in the pipeline")
    parser.add_argument("--count", type=str, dest='countLabel',
                        help="Count and display the number of specified objects on the frame. You can enter either the name of the object or its label id (number).")
    parser.add_argument("-dev", "--deviceId", type=str,
                        help="DepthAI MX id of the device to connect to. Use the word 'list' to show all devices and exit.")
    parser.add_argument('-bandw', '--bandwidth', type=str, default="auto", choices=["auto", "low", "high"], help="Force bandwidth mode. \n"
                                                                                                                 "If set to \"high\", the output streams will stay uncompressed\n"
                                                                                                                 "If set to \"low\", the output streams will be MJPEG-encoded\n"
                                                                                                                 "If set to \"auto\" (default), the optimal bandwidth will be selected based on your connection type and speed")
    parser.add_argument('-usbs', '--usbSpeed', type=str, default="usb3", choices=["usb2", "usb3"], help="Force USB communication speed. Default: %(default)s")
    parser.add_argument('-enc', '--encode', type=_comaSeparated(default=30.0, cast=float), nargs="+", default=[],
                        help="Define which cameras to encode (record) \n"
                             "Format: cameraName or cameraName,encFps \n"
                             "Example: -enc left color \n"
                             "Example: -enc color right,10 left,10")
    parser.add_argument('-encout', '--encodeOutput', type=Path, default=projectRoot, help="Path to directory where to store encoded files. Default: %(default)s")
    parser.add_argument('-xls', '--xlinkChunkSize', type=int, help="Specify XLink chunk size")
    parser.add_argument('-camo', '--cameraOrientation', type=_comaSeparated(default="AUTO", cast=orientationCast), nargs="+", default=[],
                        help=("Define cameras orientation (available: {}) \n"
                             "Format: camera_name,camera_orientation \n"
                             "Example: -camo color,ROTATE_180_DEG right,ROTATE_180_DEG left,ROTATE_180_DEG").format(', '.join(orientationChoices))
                        )
    parser.add_argument("--cameraControlls", action="store_true", help="Show camera configuration options in GUI and control them using keyboard")
    parser.add_argument("--cameraExposure", type=int, help="Specify camera saturation")
    parser.add_argument("--cameraSensitivity", type=int, help="Specify camera sensitivity")
    parser.add_argument("--cameraSaturation", type=checkRange(-10, 10), help="Specify image saturation")
    parser.add_argument("--cameraContrast", type=checkRange(-10, 10), help="Specify image contrast")
    parser.add_argument("--cameraBrightness", type=checkRange(-10, 10), help="Specify image brightness")
    parser.add_argument("--cameraSharpness", type=checkRange(0, 4), help="Specify image sharpness")

    return parser.parse_args()

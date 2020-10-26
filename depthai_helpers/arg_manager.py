import os
import argparse
try:
    import argcomplete
except ImportError:
    raise ImportError('\033[1;5;31m argcomplete module not found, run python3 -m pip install -r requirements.txt \033[0m')
from argcomplete.completers import ChoicesCompleter

from depthai_helpers.cli_utils import cli_print, PrintColors
import consts.resource_paths


def _get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

_stream_choices = ("metaout", "previewout", "jpegout", "left", "right", "depth", "disparity", "disparity_color",
                   "meta_d2h", "object_tracker", "rectified_left", "rectified_right", "color")
_CNN_choices = _get_immediate_subdirectories(consts.resource_paths.nn_resource_path)
_CNN2_choices = ['landmarks-regression-retail-0009', 'facial-landmarks-35-adas-0002', 'emotions-recognition-retail-0003']

def _stream_type(option):
    max_fps = None
    option_list = option.split(",")
    option_args = len(option_list)
    if option_args not in [1, 2]:
        msg_string = "{0} format is invalid. See --help".format(option)
        cli_print(msg_string, PrintColors.WARNING)
        raise ValueError(msg_string)

    transition_map = {"depth_raw" : "depth"}
    stream_name = option_list[0]
    if stream_name in transition_map:
        cli_print("Stream option " + stream_name + " is deprecated, use: " + transition_map[stream_name], PrintColors.WARNING)
        stream_name = transition_map[stream_name]

    if stream_name not in _stream_choices:
        msg_string = "{0} is not in available stream list: \n{1}".format(stream_name, _stream_choices)
        cli_print(msg_string, PrintColors.WARNING)
        raise ValueError(msg_string)

    if option_args == 1:
        stream_dict = {"name": stream_name}
    else:
        try:
            max_fps = float(option_list[1])
        except ValueError:
            msg_string = "In option: {0} {1} is not a number!".format(option, option_list[1])
            cli_print(msg_string, PrintColors.WARNING)

        stream_dict = {"name": stream_name, "max_fps": max_fps}
    return stream_dict

class CliArgs:
    args = []

    def __init__(self):
        super().__init__()

    def parse_args(self):
        epilog_text = """
        Displays video streams captured by DepthAI.

        Example usage:

        ## Show the depth stream:
        python3 test.py -s disparity_color,12
        ## Show the depth stream and NN output:
        python3 test.py -s metaout previewout,12 disparity_color,12
        """
        parser = argparse.ArgumentParser(epilog=epilog_text, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("-co", "--config_overwrite", default=None,
                            type=str, required=False,
                            help="JSON-formatted pipeline config object. This will be override defaults used in this "
                                 "script.")

        parser.add_argument("-brd", "--board", default=None, type=str,
                            help="BW1097, BW1098OBC - Board type from resources/boards/ (not case-sensitive). "
                                 "Or path to a custom .json board config. Mutually exclusive with [-fv -rfv -b -r -w]")

        parser.add_argument("-sh", "--shaves", default=None, type=int, choices=range(1,15), metavar="[1-14]",
                            help="Number of shaves used by NN.")

        parser.add_argument("-cmx", "--cmx_slices", default=None, type=int, choices=range(1,15), metavar="[1-14]",
                            help="Number of cmx slices used by NN.")

        parser.add_argument("-nce", "--NN_engines", default=None, type=int, choices=[1, 2], metavar="[1-2]",
                            help="Number of NN_engines used by NN.")

        parser.add_argument("-mct", "--model-compilation-target", default="auto",
                            type=str, required=False, choices=["auto","local","cloud"],
                            help="Compile model lcoally or in cloud?")

        parser.add_argument("-rgbr", "--rgb_resolution", default=1080, type=int, choices=[1080, 2160, 3040],
                            help="RGB cam res height: (1920x)1080, (3840x)2160 or (4056x)3040. Default: %(default)s")

        parser.add_argument("-rgbf", "--rgb_fps", default=30.0, type=float,
                            help="RGB cam fps: max 118.0 for H:1080, max 42.0 for H:2160. Default: %(default)s")

        parser.add_argument("-cs", "--color_scale", default=1.0, type=float,
                            help="Scale factor for 'color' stream preview window. Default: %(default)s")

        parser.add_argument("-monor", "--mono_resolution", default=720, type=int,  choices=[400, 720, 800],
                            help="Mono cam res height: (1280x)720, (1280x)800 or (640x)400 - binning. Default: %(default)s")

        parser.add_argument("-monof", "--mono_fps", default=30.0, type=float,
                            help="Mono cam fps: max 60.0 for H:720 or H:800, max 120.0 for H:400. Default: %(default)s")

        parser.add_argument("-dct", "--disparity_confidence_threshold", default=200, type=int, 
                            choices=range(0,256), metavar="[0-255]",
                            help="Disparity_confidence_threshold.")

        parser.add_argument("-med", "--stereo_median_size", default=7, type=int, choices=[0, 3, 5, 7],
                            help="Disparity / depth median filter kernel size (N x N) . 0 = filtering disabled. Default: %(default)s")

        parser.add_argument("-lrc", "--stereo_lr_check", default=False, action="store_true",
                        help="Enable stereo 'Left-Right check' feature.")
        parser.add_argument("-fv", "--field-of-view", default=None, type=float,
                            help="Horizontal field of view (HFOV) for the stereo cameras in [deg]. Default: 71.86deg.")

        parser.add_argument("-rfv", "--rgb-field-of-view", default=None, type=float,
                            help="Horizontal field of view (HFOV) for the RGB camera in [deg]. Default: 68.7938deg.")

        parser.add_argument("-b", "--baseline", default=None, type=float,
                            help="Left/Right camera baseline in [cm]. Default: 9.0cm.")

        parser.add_argument("-r", "--rgb-baseline", default=None, type=float,
                            help="Distance the RGB camera is from the Left camera. Default: 2.0cm.")

        parser.add_argument("-w", "--no-swap-lr", dest="swap_lr", default=None, action="store_false",
                            help="Do not swap the Left and Right cameras.")
        
        parser.add_argument("-e", "--store-eeprom", default=False, action="store_true",
                            help="Store the calibration and board_config (fov, baselines, swap-lr) in the EEPROM onboard")
        
        parser.add_argument("--clear-eeprom", default=False, action="store_true",
                            help="Invalidate the calib and board_config from EEPROM")
        
        parser.add_argument("-o", "--override-eeprom", default=False, action="store_true",
                            help="Use the calib and board_config from host, ignoring the EEPROM data if programmed")
        
        parser.add_argument("-dev", "--device-id", default="", type=str,
                            help="USB port number for the device to connect to. Use the word 'list' to show all devices "
                                 "and exit.")
        parser.add_argument("-debug", "--dev_debug", nargs="?", default=None, const='', type=str, required=False,
                            help="Used by board developers for debugging. Can take parameter to device binary")
        parser.add_argument("-fusb2", "--force_usb2", default=None, action="store_true",
                            help="Force usb2 connection")
        
        parser.add_argument("-cnn", "--cnn_model", default="mobilenet-ssd", type=str, choices=_CNN_choices,
                            help="Cnn model to run on DepthAI")
        
        parser.add_argument("-cnn2", "--cnn_model2", default="", type=str, choices=_CNN2_choices,
                            help="Cnn model to run on DepthAI for second-stage inference")
        
        parser.add_argument('-cam', "--cnn_camera", default='rgb',
                            choices=['rgb', 'left', 'right', 'left_right', 'rectified_left', 'rectified_right', 'rectified_left_right'],
                            help='Choose camera input for CNN (default: %(default)s)')
        
        parser.add_argument("-dd", "--disable_depth", default=False, action="store_true",
                            help="Disable depth calculation on CNN models with bounding box output")
        
        parser.add_argument("-bb", "--draw-bb-depth", default=False, action="store_true",
                            help="Draw the bounding boxes over the left/right/depth* streams")
        
        parser.add_argument("-ff", "--full-fov-nn", default=False, action="store_true",
                            help="Full RGB FOV for NN, not keeping the aspect ratio")

        parser.add_argument("-sync", "--sync-video-meta", default=False, action="store_true",
                            help="Synchronize 'previewout' and 'metaout' streams")

        parser.add_argument("-seq", "--sync-sequence-numbers", default=False, action="store_true",
                            help="Synchronize sequence numbers for all packets. Experimental")

        parser.add_argument("-u", "--usb-chunk-KiB", default=64, type=int,
                            help="USB transfer chunk on device. Higher (up to megabytes) "
                            "may improve throughput, or 0 to disable chunking. Default: %(default)s")

        parser.add_argument("-fw", "--firmware", default=None, type=str,
                            help="Commit hash for custom FW, downloaded from Artifactory")

        parser.add_argument("-vv", "--verbose", default=False, action="store_true",
                        help="Verbose, print info about received packets.")
        parser.add_argument("-s", "--streams",
                            nargs="+",
                            type=_stream_type,
                            dest="streams",
                            default=["metaout", "previewout"],
                            help=("Define which streams to enable \n"
                                  "Format: stream_name or stream_name,max_fps \n"
                                  "Example: -s metaout previewout \n"
                                  "Example: -s metaout previewout,10 depth_sipp,10"))\
                            .completer=ChoicesCompleter(_stream_choices)
        
        parser.add_argument("-v", "--video", default=None, type=str, required=False,
                            help="Path where to save video stream (existing file will be overwritten)")
        
        parser.add_argument("-pcl", "--pointcloud", default=False, action="store_true",
                        help="Convert Depth map image to point clouds")

        parser.add_argument("-mesh", "--use_mesh", default=False, action="store_true",
                        help="use mesh for rectification of the stereo images")

        parser.add_argument("-mirror_rectified", "--mirror_rectified", default='true', choices=['true', 'false'],
                        help="Normally, rectified_left/_right are mirrored for Stereo engine constraints. "
                             "If false, disparity/depth will be mirrored instead. Default: true")
        argcomplete.autocomplete(parser)

        options = parser.parse_args()
        any_options_set = any([options.field_of_view, options.rgb_field_of_view, options.baseline, options.rgb_baseline,
                               options.swap_lr])
        if (options.board is not None) and any_options_set:
            parser.error("[-brd] is mutually exclusive with [-fv -rfv -b -r -w]")

        # Set some defaults after the above check
        if not options.board:
            options.field_of_view = 71.86
            options.rgb_field_of_view = 68.7938
            options.baseline = 9.0
            options.rgb_baseline = 2.0
            options.swap_lr = True

        return options

#!/usr/bin/env python3

import argparse
from enum import Enum


class PrintColors(Enum):
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    WARNING = "\033[1;5;31m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def cli_print(msg, print_color):
    """
    Prints to console with input print color type
    """
    if not isinstance(print_color, PrintColors):
        raise ValueError("Must use PrintColors type in cli_print")
    print("{0}{1}{2}".format(print_color.value, msg, PrintColors.ENDC.value))


def parse_args():
    epilog_text = """
    Displays video streams captured by DepthAI.

    Example usage:

    ## Show the depth stream:
    python3 test.py -s depth_sipp,12
    ## Show the depth stream and NN output:
    python3 test.py -s metaout previewout,12 depth_sipp,12
    """
    parser = argparse.ArgumentParser(epilog=epilog_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-co", "--config_overwrite", default=None,
                        type=str, required=False,
                        help="JSON-formatted pipeline config object. This will be override defaults used in this "
                             "script.")
    parser.add_argument("-brd", "--board", default=None, type=str,
                        help="BW1097, BW1098OBC - Board type from resources/boards/ (not case-sensitive). "
                             "Or path to a custom .json board config. Mutually exclusive with [-fv -rfv -b -r -w]")
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
    parser.add_argument("-debug", "--dev_debug", default=None, action="store_true",
                        help="Used by board developers for debugging.")
    parser.add_argument("-fusb2", "--force_usb2", default=None, action="store_true",
                        help="Force usb2 connection")
    parser.add_argument("-cnn", "--cnn_model", default="mobilenet-ssd", type=str,
                        help="Cnn model to run on DepthAI")
    parser.add_argument("-dd", "--disable_depth", default=False, action="store_true",
                        help="Disable depth calculation on CNN models with bounding box output")
    parser.add_argument("-bb", "--draw-bb-depth", default=False, action="store_true",
                        help="Draw the bounding boxes over the left/right/depth* streams")
    parser.add_argument("-ff", "--full-fov-nn", default=False, action="store_true",
                        help="Full RGB FOV for NN, not keeping the aspect ratio")
    parser.add_argument("-s", "--streams",
                        nargs="+",
                        type=stream_type,
                        dest="streams",
                        default=["metaout", "previewout"],
                        help=("Define which streams to enable \n"
                              "Format: stream_name or stream_name,max_fps \n"
                              "Example: -s metaout previewout \n"
                              "Example: -s metaout previewout,10 depth_sipp,10"))
    parser.add_argument("-v", "--video", default=None, type=str, required=False,
                        help="Path where to save video stream (existing file will be overwritten)")

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


def stream_type(option):
    max_fps = None
    option_list = option.split(",")
    option_args = len(option_list)
    if option_args not in [1, 2]:
        msg_string = "{0} format is invalid. See --help".format(option)
        cli_print(msg_string, PrintColors.WARNING)
        raise ValueError(msg_string)

    stream_choices = ["metaout", "previewout", "jpegout", "left", "right", "depth_sipp", "disparity", "depth_color_h", "meta_d2h", "object_tracker"]
    stream_name = option_list[0]
    if stream_name not in stream_choices:
        msg_string = "{0} is not in available stream list: \n{1}".format(stream_name, stream_choices)
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

#!/usr/bin/env python3

import argparse
from argparse import ArgumentParser


class PrintColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    WARNING = '\033[1;5;31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_args():
    epilog_text = '''
    Displays video streams captured by DepthAI.

    Example usage:

    ## Show the depth stream:
    python3 test.py -s depth_sipp,12
    ## Show the depth stream and NN output:
    python3 test.py -s metaout previewout,12 depth_sipp,12
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-co", "--config_overwrite", default=None,
                        type=str, required=False,
                        help="JSON-formatted pipeline config object. This will be override defaults used in this script.")
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
    parser.add_argument("-e", "--store-eeprom", default=False, action='store_true',
                        help="Store the calibration and board_config (fov, baselines, swap-lr) in the EEPROM onboard")
    parser.add_argument("--clear-eeprom", default=False, action='store_true',
                        help="Invalidate the calib and board_config from EEPROM")
    parser.add_argument("-o", "--override-eeprom", default=False, action='store_true',
                        help="Use the calib and board_config from host, ignoring the EEPROM data if programmed")
    parser.add_argument("-dev", "--device-id", default='', type=str,
                        help="USB port number for the device to connect to. Use the word 'list' to show all devices and exit.")
    parser.add_argument("-debug", "--dev_debug", default=None, action='store_true',
                        help="Used by board developers for debugging.")
    parser.add_argument("-fusb2", "--force_usb2", default=None, action='store_true',
                        help="Force usb2 connection")
    parser.add_argument("-cnn", "--cnn_model", default='mobilenet-ssd', type=str,
                        help="Cnn model to run on DepthAI")
    parser.add_argument("-dd", "--disable_depth", default=False,  action='store_true',
                        help="Disable depth calculation on CNN models with bounding box output")
    parser.add_argument("-bb", "--draw-bb-depth", default=False,  action='store_true',
                        help="Draw the bounding boxes over the left/right/depth* streams")
    parser.add_argument("-ff", "--full-fov-nn", default=False,  action='store_true',
                        help="Full RGB FOV for NN, not keeping the aspect ratio")
    parser.add_argument("-s", "--streams",
                        nargs='+',
                        type=stream_type,
                        dest='streams',
                        default=['metaout', 'previewout'],
                        help="Define which streams to enable \
                        Format: stream_name or stream_name,max_fps \
                        Example: -s metaout previewout \
                        Example: -s metaout previewout,10 depth_sipp,10")
    parser.add_argument("-v", "--video", default=None, type=str, required=False, help="Path where to save video stream (existing file will be overwritten)")

    options = parser.parse_args()

    if (options.board is not None) and ((options.field_of_view     is not None)
                                     or (options.rgb_field_of_view is not None)
                                     or (options.baseline          is not None)
                                     or (options.rgb_baseline      is not None)
                                     or (options.swap_lr           is not None)):
        parser.error("[-brd] is mutually exclusive with [-fv -rfv -b -r -w]")

    # Set some defaults after the above check
    if options.field_of_view     is None: options.field_of_view = 71.86
    if options.rgb_field_of_view is None: options.rgb_field_of_view = 68.7938
    if options.baseline          is None: options.baseline = 9.0
    if options.rgb_baseline      is None: options.rgb_baseline = 2.0
    if options.swap_lr           is None: options.swap_lr = True

    return options

def stream_type(option):
    option_list = option.split(',')
    option_args = len(option_list)
    if option_args not in [1,2]:
        print(PrintColors.WARNING + option + " format is invalid. See --help" + PrintColors.ENDC)
        raise ValueError

    stream_choices=['metaout', 'previewout', 'left', 'right', 'depth_sipp', 'disparity', 'depth_color_h', 'meta_d2h']
    stream_name = option_list[0]
    if stream_name not in stream_choices:
        print(PrintColors.WARNING + stream_name + " is not in available stream list: \n" + str(stream_choices) + PrintColors.ENDC)
        raise ValueError

    if(option_args == 1):
        stream_dict = {'name': stream_name}
    else:
        try:
            max_fps = float(option_list[1])
        except:
            print(PrintColors.WARNING + "In option: " + str(option) + " " + option_list[1] + " is not a number!" + PrintColors.ENDC)

        stream_dict = {'name': stream_name, "max_fps": max_fps}
    return stream_dict

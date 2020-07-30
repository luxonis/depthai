#!/usr/bin/env python3

from enum import Enum
import os
import consts.resource_paths

def _get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

_stream_choices = ("metaout", "previewout", "jpegout", "left", "right", "depth_raw", "disparity", "disparity_color", "meta_d2h", "object_tracker")
_CNN_choices = _get_immediate_subdirectories(consts.resource_paths.nn_resource_path)

def _stream_type(option):
    max_fps = None
    option_list = option.split(",")
    option_args = len(option_list)
    if option_args not in [1, 2]:
        msg_string = "{0} format is invalid. See --help".format(option)
        cli_print(msg_string, PrintColors.WARNING)
        raise ValueError(msg_string)

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


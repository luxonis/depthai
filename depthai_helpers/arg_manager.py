import argparse
from depthai_helpers.cli_utils import cli_print, PrintColors


def stream_type(option):
    max_fps = None
    option_list = option.split(",")
    option_args = len(option_list)
    if option_args not in [1, 2]:
        msg_string = "{0} format is invalid. See --help".format(option)
        cli_print(msg_string, PrintColors.WARNING)
        raise ValueError(msg_string)

    stream_choices = ["metaout", "video", "jpegout", "previewout", "left", "right", "depth_sipp", "disparity", "depth_color_h", "meta_d2h", "object_tracker"]
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

def disparity_ct_type(value):
    try: 
        value = int(value)
    except ValueError:
        raise ValueError("Confidence threshold should be INT")
    if value < 0 or value > 255:
        msg_string = "{0} disparity confidence threshold is not in interval [0,255]. See --help".format(value)
        cli_print(msg_string, PrintColors.WARNING)
        raise ValueError(msg_string)
    
    return value


class Arg():
    shortName = None
    longName = None
    required = None
    default = None
    type=None
    help=None
    action=None
    choices=None
    dest=None
    nargs=None
    const=None

    #ROS2 specific
    ros2Type=str

    def __init__(self, sn=None, ln=None, default=None, type=None, help=None, required = None, action=None, choices=None, dest=None, nargs=None, const=None, ros2Type=str):
        self.shortName = sn
        self.longName = ln
        self.default = default
        self.type = type
        self.help = help
        self.required = required
        self.action = action
        self.choices = choices
        self.dest = dest
        self.nargs = nargs
        self.const = const

        self.ros2Type = ros2Type

class SharedArgs:
    args = []

    def __init__(self):
        self.args.append(Arg("-co", "--config_overwrite", default=None, type="str", required=False,
            help="JSON-formatted pipeline config object. This will be override defaults used in this "
                "script.", ros2Type="str"))
        self.args.append(Arg("-brd", "--board", default=None, type="str",
            help="BW1097, BW1098OBC - Board type from resources/boards/ (not case-sensitive). "
                 "Or path to a custom .json board config. Mutually exclusive with [-fv -rfv -b -r -w]", ros2Type="str"))
        self.args.append(Arg("-sh", "--shaves", default=None, type="int",
            help="Number of shaves used by NN.", ros2Type="int"))
        self.args.append(Arg("-cmx", "--cmx_slices", default=None, type="int",
            help="Number of cmx slices used by NN.", ros2Type="int"))
        self.args.append(Arg("-nce", "--NN_engines", default=None, type="int",
            help="Number of NN_engines used by NN.", ros2Type="int"))
        self.args.append(Arg("-rgbr", "--rgb_resolution", default=1080, type="int",
            help="RGB cam res height: (1920x)1080, (3840x)2160 or (4056)x3040. Default: %(default)s", ros2Type="int"))
        self.args.append(Arg("-rgbf", "--rgb_fps", default=30.0, type="float",
            help="RGB cam fps: max 118.0 for H:1080, max 42.0 for H:2160. Default: %(default)s", ros2Type="float"))
        self.args.append(Arg("-monor", "--mono_resolution", default=720, type="int",
            help="Mono cam res height: (1280x)720, (1280x)800 or (640x)400 - binning. Default: %(default)s", ros2Type="int"))
        self.args.append(Arg("-monof", "--mono_fps", default=30.0, type="float",
            help="Mono cam fps: max 60.0 for H:720 or H:800, max 120.0 for H:400. Default: %(default)s", ros2Type="float"))
        self.args.append(Arg("-dct", "--disparity_confidence_threshold", default=200, type="disparity_ct_type",
            help="Disparity_confidence_threshold.", ros2Type="int"))
        self.args.append(Arg("-fv", "--field-of-view", default=None, type="float",
            help="Horizontal field of view (HFOV) for the stereo cameras in [deg]. Default: 71.86deg.", ros2Type="float"))
        self.args.append(Arg("-rfv", "--rgb-field-of-view", default=None, type="float",
            help="Horizontal field of view (HFOV) for the RGB camera in [deg]. Default: 68.7938deg.", ros2Type="float"))
        self.args.append(Arg("-b", "--baseline", default=None, type="float",
            help="Left/Right camera baseline in [cm]. Default: 9.0cm.", ros2Type="float"))
        self.args.append(Arg("-r", "--rgb-baseline", default=None, type="float",
            help="Distance the RGB camera is from the Left camera. Default: 2.0cm.", ros2Type="float"))
        self.args.append(Arg("-w", "--no-swap-lr", dest="swap_lr", default=None, action="store_false",
            help="Do not swap the Left and Right cameras.", ros2Type="bool"))
        self.args.append(Arg("-e", "--store-eeprom", default=False, action="store_true",
            help="Store the calibration and board_config (fov, baselines, swap-lr) in the EEPROM onboard", ros2Type="bool"))
        self.args.append(Arg(ln="--clear-eeprom", default=False, action="store_true",
            help="Invalidate the calib and board_config from EEPROM", ros2Type="bool"))
        self.args.append(Arg("-o", "--override-eeprom", default=False, action="store_true",
            help="Use the calib and board_config from host, ignoring the EEPROM data if programmed", ros2Type="bool"))
        self.args.append(Arg("-dev", "--device-id", default="", type="str",
            help="USB port number for the device to connect to. Use the word 'list' to show all devices "
                 "and exit.", ros2Type="str"))
        self.args.append(Arg("-debug", "--dev_debug", nargs="?", default=None, const='', type="str", required=False,
            help="Used by board developers for debugging. Can take parameter to device binary", ros2Type="str"))
        self.args.append(Arg("-fusb2", "--force_usb2", default=None, action="store_true",
            help="Force usb2 connection", ros2Type="bool"))
        self.args.append(Arg("-cnn", "--cnn_model", default="mobilenet-ssd", type="str",
            help="Cnn model to run on DepthAI", ros2Type="str"))
        self.args.append(Arg("-cnn2", "--cnn_model2", default="", type="str",
            help="Cnn model to run on DepthAI for second-stage inference", ros2Type="str"))
        self.args.append(Arg('-cam', "--cnn_camera", default='rgb', choices=['rgb', 'left', 'right', 'left_right'],
            help='Choose camera input for CNN (default: %(default)s)', ros2Type="strArray"))
        self.args.append(Arg("-dd", "--disable_depth", default=False, action="store_true",
            help="Disable depth calculation on CNN models with bounding box output", ros2Type="bool"))
        self.args.append(Arg("-bb", "--draw-bb-depth", default=False, action="store_true",
            help="Draw the bounding boxes over the left/right/depth* streams", ros2Type="bool"))
        self.args.append(Arg("-ff", "--full-fov-nn", default=False, action="store_true",
            help="Full RGB FOV for NN, not keeping the aspect ratio", ros2Type="bool"))
        self.args.append(Arg("-s", "--streams",
                            nargs="+",
                            type="stream_type",
                            dest="streams",
                            default=["metaout", "previewout"],
                            help=("Define which streams to enable \n"
                                  "Format: stream_name or stream_name,max_fps \n"
                                  "Example: -s metaout previewout \n"
                                  "Example: -s metaout previewout,10 depth_sipp,10"), ros2Type="strArray"))
        self.args.append(Arg("-v", "--video", default=None, type="str", required=False,
                            help="Path where to save video stream (existing file will be overwritten)", ros2Type="str"))


class CliArgs(SharedArgs):
    args = []

    def __init__(self):
        super().__init__()

    def parse_args(self):
        epilog_text = """
        Displays video streams captured by DepthAI.

        Example usage:

        ## Show the depth stream:
        python3 test.py -s depth_sipp,12
        ## Show the depth stream and NN output:
        python3 test.py -s metaout previewout,12 depth_sipp,12
        """
        myArgs = SharedArgs()

        parser = argparse.ArgumentParser(epilog=epilog_text, formatter_class=argparse.RawDescriptionHelpFormatter)

        for arg in myArgs.args:
            evalStr = "parser.add_argument("
            if arg.shortName is not None:
                evalStr += "\""+arg.shortName+"\","
            if arg.longName is not None:
                evalStr += "\""+arg.longName+"\","

            if arg.default is not None:
                if type(arg.default) is str:
                    evalStr += "default=\""+arg.default+"\","
                else:
                    evalStr += "default="+str(arg.default)+","
            else:
                evalStr += "default=None,"

            if arg.required is not None:
                evalStr += "required="+str(arg.required)+","
            if arg.type is not None:
                evalStr += "type="+arg.type+","
            if arg.help is not None:
                evalStr += "help=\"\"\""+arg.help+"\"\"\","
            if arg.action is not None:
                evalStr += "action=\""+arg.action+"\","
            if arg.choices is not None:
                evalStr += "choices="+str(arg.choices)+","
            if arg.dest is not None:
                evalStr += "dest=\""+arg.dest+"\","
            if arg.nargs is not None:
                evalStr += "nargs=\""+arg.nargs+"\","
            if arg.const is not None:
                evalStr += "const=\""+arg.const+"\","
            evalStr = evalStr.rstrip(',')
            evalStr += ")"
            eval(evalStr)

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

from pathlib import Path


def relative_to_abs_path(relative_path):
    dirname = Path(__file__).parent
    try:
        return str((dirname / relative_path).resolve())
    except FileNotFoundError:
        return None

prefix                = relative_to_abs_path('../resources/')+"/"
device_cmd_fpath      = relative_to_abs_path('../depthai.cmd')
device_usb2_cmd_fpath = relative_to_abs_path('../depthai_usb2.cmd')
boards_dir_path       = relative_to_abs_path('../resources/boards') + "/"
custom_calib_fpath    = relative_to_abs_path('../resources/depthai.calib')
left_mesh_fpath       = relative_to_abs_path('../resources/mesh_left.calib')
right_mesh_fpath      = relative_to_abs_path('../resources/mesh_right.calib')

right_map_x_fpath     = relative_to_abs_path('../resources/map_x_right.calib')
right_map_y_fpath     = relative_to_abs_path('../resources/map_y_right.calib')
left_map_x_fpath      = relative_to_abs_path('../resources/map_x_left.calib')
left_map_y_fpath      = relative_to_abs_path('../resources/map_y_left.calib')

nn_resource_path      = relative_to_abs_path('../resources/nn')+"/"
blob_fpath            = relative_to_abs_path('../resources/nn/mobilenet-ssd/mobilenet-ssd.blob')
blob_config_fpath     = relative_to_abs_path('../resources/nn/mobilenet-ssd/mobilenet-ssd.json')
tests_functional_path = relative_to_abs_path('../testsFunctional/') + "/"


if custom_calib_fpath is not None and Path(custom_calib_fpath).exists():
    calib_fpath = custom_calib_fpath
    print("Using Custom Calibration File: depthai.calib")
else:
    calib_fpath = ''
    print("No calibration file. Using Calibration Defaults.")

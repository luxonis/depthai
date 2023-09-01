from pathlib import Path
from string import Template
from typing import Union, Tuple

import depthai as dai

from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.components.camera_helper import get_resolution_size


def control_camera_with_nn(
        pipeline: dai.Pipeline,
        camera_control: dai.Node.Input,
        nn_output: dai.Node.Output,
        resize_mode: ResizeMode,
        resolution: Union[dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution],
        nn_size: Tuple[int, int],
        af: bool,
        ae: bool,
        debug: bool
):
    sensor_resolution = get_resolution_size(resolution)
    # width / height (old ar)
    sensor_ar = sensor_resolution[0] / sensor_resolution[1]
    # NN ar (new ar)
    nn_ar = nn_size[0] / nn_size[1]

    if resize_mode == ResizeMode.LETTERBOX:
        padding = (sensor_ar - nn_ar) / 2
        if padding > 0:
            init = f"xmin = 0; ymin = {-padding}; xmax = 1; ymax = {1 + padding}"
        else:
            init = f"xmin = {padding}; ymin = 0; xmax = {1 - padding}; ymax = 1"
    elif resize_mode in [ResizeMode.CROP, ResizeMode.FULL_CROP]:
        cropping = (1 - (nn_ar / sensor_ar)) / 2
        if cropping < 0:
            init = f"xmin = 0; ymin = {-cropping}; xmax = 1; ymax = {1 + cropping}"
        else:
            init = f"xmin = {cropping}; ymin = 0; xmax = {1 - cropping}; ymax = 1"
    else:  # Stretch
        init = f"xmin=0; ymin=0; xmax=1; ymax=1"

    resize_str = f"new_xmin=xmin+width*det.xmin; new_ymin=ymin+height*det.ymin; new_xmax=xmin+width*det.xmax; new_ymax=ymin+height*det.ymax;"
    denormalize = f"startx=int(new_xmin*{sensor_resolution[0]}); starty=int(new_ymin*{sensor_resolution[1]}); new_width=int((new_xmax-new_xmin)*{sensor_resolution[0]}); new_height=int((new_ymax-new_ymin)*{sensor_resolution[1]});"
    control_str = ''
    if ae:
        control_str += f"control.setAutoExposureRegion(startx, starty, new_width, new_height);"
    if af:
        control_str += f"control.setAutoFocusRegion(startx, starty, new_width, new_height);"

    script_node = pipeline.create(dai.node.Script)
    script_node.setProcessor(dai.ProcessorType.LEON_CSS)  # More stable

    with open(Path(__file__).parent / 'template_control_cam_with_nn.py', 'r') as file:
        code = Template(file.read()).substitute(
            DEBUG='' if debug else '#',
            INIT=init,
            RESIZE=resize_str,
            DENORMALIZE=denormalize,
            CONTROL=control_str
        )
        script_node.setScript(code)

        if debug:
            print(code)

    # Node linking:
    # NN output -> Script -> Camera input
    nn_output.link(script_node.inputs['detections'])
    script_node.outputs['control'].link(camera_control)

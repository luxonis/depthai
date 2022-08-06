import depthai as dai
from typing import Union, Tuple, Optional, Dict, Any
from .camera_helper import setCameraControl

def rgbResolution(resolution: Union[str, dai.ColorCameraProperties.SensorResolution]) -> dai.ColorCameraProperties.SensorResolution:
    """
    Parses Color camera resolution based on the string
    """
    if isinstance(resolution, dai.ColorCameraProperties.SensorResolution):
        return resolution

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

def monoResolution(resolution: Union[str, dai.MonoCameraProperties.SensorResolution]) -> dai.MonoCameraProperties.SensorResolution:
    """
    Parses Mono camera resolution based on the string
    """
    if isinstance(resolution, dai.MonoCameraProperties.SensorResolution):
        return resolution

    resolution = str(resolution).upper()
    if resolution == '800' or resolution == '800P':
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    elif resolution == '720' or resolution == '720P':
        return dai.MonoCameraProperties.SensorResolution.THE_720_P
    elif resolution == '480' or resolution == '480P':
        return dai.MonoCameraProperties.SensorResolution.THE_480_P
    else: # Default
        return dai.MonoCameraProperties.SensorResolution.THE_400_P

def parseResolution(
    camera: Union[dai.node.ColorCamera, dai.node.MonoCamera],
    resolution: Union[str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution]
    ):
    if isinstance(camera, dai.node.ColorCamera):
        return rgbResolution(resolution)
    elif isinstance(camera, dai.node.MonoCamera):
        return monoResolution(resolution)
    else:
        raise ValueError("camera must be either MonoCamera or ColorCamera!")

def parseSize(size: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(size, Tuple):
        return size
    elif isinstance(size, str):
        vals = size.split('x')
        if len(vals) != 2: raise ValueError("Size must be in format '[width]x[height]'!")
        return Tuple[int(vals[0]), int(vals[1])]
    else:
        raise ValueError("Size typle not supported!")


def parseColorCamControl(options: Dict[str, Any], cam: dai.node.ColorCamera):
    setCameraControl(cam.initialControl,
                     options.get('manualFocus', None),
                     options.get('afMode', None),
                     options.get('awbMode', None),
                     options.get('sceneMode', None),
                     options.get('antiBandingMode', None),
                     options.get('effectMode', None),
                     options.get('sharpness', None),
                     options.get('lumaDenoise', None),
                     options.get('chromaDenoise', None),
                     )
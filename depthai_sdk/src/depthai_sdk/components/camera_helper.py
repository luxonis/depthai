import math
import depthai as dai
from typing import *

class ImageSensor:
    name: str
    resolution: Union[
        dai.ColorCameraProperties.SensorResolution,
        dai.MonoCameraProperties.SensorResolution
    ]
    type: dai.node

    def __eq__(self, other):
        return self.name == other

    def __init__(self,
                 name: str,
                 resolution: str,
                 type: str):
        from .parser import parseResolution
        self.name = name
        self.type = dai.node.ColorCamera if type == 'color' else dai.node.MonoCamera
        self.resolution = parseResolution(self.type, resolution)


cameraSensors: List[ImageSensor] = [
    ImageSensor('IMX378', '12mp', 'color'),
    ImageSensor('OV9282', '800P', 'mono'),
    ImageSensor('OV9782', '800P', 'color'),
    ImageSensor('OV9281', '800P', 'color'),
    ImageSensor('IMX214', '13mp', 'color'),
    ImageSensor('OV7750', '480P', 'mono'),
    ImageSensor('OV7251', '480P', 'mono'),
    ImageSensor('IMX477', '12mp', 'color'),
    ImageSensor('IMX577', '12mp', 'color'),
    ImageSensor('AR0234', '1200P', 'color'),
    ImageSensor('IMX582', '48mp', 'color'),
]

cameraResolutions: Dict[Any, Tuple[int, int]] = {
    dai.ColorCameraProperties.SensorResolution.THE_13_MP: (4208, 3120),
    dai.ColorCameraProperties.SensorResolution.THE_12_MP: (4056, 3040),
    dai.ColorCameraProperties.SensorResolution.THE_4_K: (3840, 2160),
    dai.ColorCameraProperties.SensorResolution.THE_1080_P: (1920, 1080),
    dai.ColorCameraProperties.SensorResolution.THE_800_P: (1280, 800),
    dai.ColorCameraProperties.SensorResolution.THE_720_P: (1280, 720),

    dai.MonoCameraProperties.SensorResolution.THE_800_P: (1280, 800),
    dai.MonoCameraProperties.SensorResolution.THE_720_P: (1280, 720),
    dai.MonoCameraProperties.SensorResolution.THE_480_P: (640, 480),
    dai.MonoCameraProperties.SensorResolution.THE_400_P: (640, 400),
}



def availableIspScales() -> List[Tuple[int, Tuple[int, int]]]:
    """
    Calculates all supported
    @rtype: List[ratio, [Numerator, Denominator]]
    """
    lst = []
    for n in range(1, 16 + 1):
        for d in range(n, 63 + 1):
            # Chroma needs 2x extra downscaling
            if d < 32 or n % 2 == 0:
                # Only if irreducible
                if math.gcd(n, d) == 1:
                    lst.append((n / d, (n, d)))
    lst.sort(reverse=True)
    return lst


def getClosestIspScale(camResolution: Tuple[int, int],
                       width: Optional[int] = None,
                       height: Optional[int] = None,
                       videoEncoder: bool = False,
                       encoderFlag: bool = True,
                       ) -> List[int]:
    """
    Provides the closest ISP scaling values to either specified width or height.
    @param camResolution: Resolution (W, H) of the ColorCamera
    @param width: Desired width after ISP scaling. Conflicts with height
    @param height: Desired height after ISP scaling. Conflicts with width
    @param videoEncoder: If we want to stream ISP output directly into VideoEncoder (take into account its limitations).
    Width and height scaling values won't be the same (ignore aspect ratio), as otherwise only 5 ISP scaling options are
    available.
    @param encoderFlag: Not for user. Flag to avoid infinite looping.
    @return ISP scaling values (list of 4 ints)
    """
    if width and height:
        raise ValueError("You have to specify EITHER width OR height to calculate desired ISP scaling options!")
    if not width and not height:
        raise ValueError("You have to provide width or height calculate desired ISP scaling options!")

    minError = 99999
    ispScale: List[int] = None
    for ratio, (n, d) in availableIspScales():
        newW = int((camResolution[0] * n - 1) / d + 1)
        newH = int((camResolution[1] * n - 1) / d + 1)

        if (videoEncoder and
                (width and newW % 32 != 0 or
                 height and newH % 8 != 0)):
            continue  # ISP output size isn't supported by VideoEncoder

        # Currently, new ISP width must be divisible by 2. FW engineers are looking into it.
        if newW % 2 != 0:
            continue

        err = abs((newW - width) if width else (newH - height))
        if err < minError:
            ispScale = [n, d, n, d]
            minError = err

    if videoEncoder and encoderFlag:
        # Calculate the ISP scale for the other direction. Note that this means aspect ratio won't be preserved
        if width:
            hScale = getClosestIspScale(camResolution,
                                        height=int(camResolution[1] * ispScale[0] / ispScale[1]),
                                        videoEncoder=True,
                                        encoderFlag=False  # To avoid infinite loop
                                        )
            ispScale[2] = hScale[2]
            ispScale[3] = hScale[3]
        else:
            wScale = getClosestIspScale(camResolution,
                                        width=int(camResolution[0] * ispScale[0] / ispScale[1]),
                                        videoEncoder=True,
                                        encoderFlag=False  # To avoid infinite loop
                                        )
            ispScale[0] = wScale[0]
            ispScale[1] = wScale[1]

    return ispScale


def setCameraControl(control: dai.CameraControl,
                     manualFocus: Optional[int] = None,
                     afMode: Optional[dai.CameraControl.AutoFocusMode] = None,
                     awbMode: Optional[dai.CameraControl.AutoWhiteBalanceMode] = None,
                     sceneMode: Optional[dai.CameraControl.SceneMode] = None,
                     antiBandingMode: Optional[dai.CameraControl.AntiBandingMode] = None,
                     effectMode: Optional[dai.CameraControl.EffectMode] = None,
                     sharpness: Optional[int] = None,
                     lumaDenoise: Optional[int] = None,
                     chromaDenoise: Optional[int] = None,

                     ):
    if manualFocus is not None:
        control.setManualFocus(manualFocus)
    if afMode is not None:
        control.setAutoFocusMode(afMode)
    if awbMode is not None:
        control.setAutoWhiteBalanceMode(awbMode)
    if sceneMode is not None:
        control.setSceneMode(sceneMode)
    if antiBandingMode is not None:
        control.setAntiBandingMode(antiBandingMode)
    if effectMode is not None:
        control.setEffectMode(effectMode)
    if sharpness is not None:
        control.setSharpness(sharpness)
    if lumaDenoise is not None:
        control.setLumaDenoise(lumaDenoise)
    if chromaDenoise is not None:
        control.setChromaDenoise(chromaDenoise)

    # TODO: Add contrast, exposure compensation, brightness, manual exposure, and saturation

def cameraSensorType(sensorName: str) -> dai.node:
    """
    Gets camera sensor type from it's name, either MonoCamera or ColorCamera.
    @param sensorName: Name of the camera sensor
    @return: dai.node.MonoCamera or dai.node.ColorCamera
    """
    return cameraSensors[cameraSensors.index(sensorName.upper())].type

def cameraSensorResolution(sensorName: str
    ) -> Union[dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution]:
    """
    Gets camera sensor type from it's name, either MonoCamera or ColorCamera.
    @param sensorName: Name of the camera sensor
    @return: dai.node.MonoCamera or dai.node.ColorCamera
    """
    return cameraSensors[cameraSensors.index(sensorName.upper())].resolution

def cameraSensorResolutionSize(sensorName: str) -> Tuple[int, int]:
    res = cameraSensorResolution(sensorName)
    return cameraResolutions[res]

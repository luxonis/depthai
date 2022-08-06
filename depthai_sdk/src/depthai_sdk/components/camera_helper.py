import math
import depthai as dai
from typing import *


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
    if width and height: raise ValueError(
        "You have to specify EITHER width OR height to calculate desired ISP scaling options!")
    if not width and not height: raise ValueError(
        "You have to provide width or height calculate desired ISP scaling options!")

    minError = 99999
    ispScale: List[int] = None
    for ratio, (n, d) in availableIspScales():
        newW = int((camResolution[0] * n - 1) / d + 1)
        newH = int((camResolution[1] * n - 1) / d + 1)

        if (videoEncoder and
                (width and newW % 32 != 0 or
                 height and newH % 8 != 0)):
            continue  # ISP output size isn't supported by VideoEncoder

        if width:
            err = abs(newW - width)
            if err < minError:
                ispScale = [n, d, n, d]
                minError = err
        else:
            err = abs(newH - height)
            if err < minError:
                ispScale = [n, d, n, d]
                minError = err

    if videoEncoder and encoderFlag:
        # Calculate the ISP scale for the other direction. Note that this means aspect ratio will be ignored.
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

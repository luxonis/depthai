from .fps import *
from .previews import *
from .utils import *
from .managers import *
from .record import *
from .replay import *

from typing import Optional
import depthai as dai

from .components import *

class Camera:
    """
    TODO: Write useful comments for users

    Camera is the main abstraction class for the OAK cameras. It abstracts pipeline building, different camera permutations,
    AI model handling, visualization, user arguments, syncing, etc.

    This abstraction layer will internally use SDK Components.
    """
    _pipeline: dai.Pipeline
    _devices: List[dai.Device]
    _args = None # User defined arguments

    def __init__(self, 
        device: Optional[str] = None, # MxId / IP / USB port / "ALL"
        usb2: Optional[bool] = None, # Auto by default
        args: bool = True
        ) -> None:
        """
        Args:
            device (str, optional): OAK device we want to connect to
            args (bool, optional): Use user defined arguments when constructing the pipeline
        """

        self._devices = self._get_device(device, usb2)
        self._pipeline = dai.Pipeline()

        if args:
            am = ArgsManager()
            self._args = am.parseArgs()

    def create_camera(self,
        source: Optional[str] = None,
        name: Optional[str] = None,
        out: bool = False,
        ) -> CameraComponent:
        """
        Create Color camera
        """
        return CameraComponent(
            source,
            name,
            out,
            args = self._args
        )

    def create_nn(self,
        ) -> NNComponent:
        a = 5
        

    def _get_device(self,
        device: Optional[str],
        usb2: Optional[bool] = None) -> List[dai.Device]:
        """
        Connect to the OAK camera and return dai.Device object
        """
        if device.upper() == "ALL":
            # Connect to all available cameras
            a = 1
        if usb2 is not None:
            return dai.Device(
                version = dai.OpenVINO.VERSION_2021_4,
                deviceInfo = getDeviceInfo(device),
                usb2Mode = usb2
                )
        else:
            return dai.Device(
                version = dai.OpenVINO.VERSION_2021_4,
                deviceInfo = getDeviceInfo(device),
                maxUsbSpeed = dai.UsbSpeed.SUPER_PLUS
                )

    def configPipeline(self,
        xlinkChunk: Optional[int] = None,
        calib: Optional[dai.CalibrationHandler] = None,
        tuningBlob: Optional[str] = None,
        ) -> None:
        if xlinkChunk is not None:
            self._pipeline.setXLinkChunkSize(xlinkChunk)
        if calib is not None:
            self._pipeline.setCalibrationData(calib)
        if tuningBlob is not None:
            self._pipeline.setCameraTuningBlobPath(tuningBlob)

    @property
    def pipeline(self) -> dai.Pipeline:
        return self._pipeline

    @property
    def device(self) -> dai.Device:
        return self._devices[0]
    
    @property
    def devices(self) -> List[dai.Device]:
        return self._devices
        
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

    This abstraction layer will internally use SDK Managers.
    """
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

        self.device = self._get_device(device, usb2)
        self.pipeline = dai.Pipeline()

        if args:
            am = ArgsManager()
            self._args = am.parseArgs()

    def create_camera(
        source: Optional[str] = None,
        name: Optional[str] = None,
        out: bool = False,
        ) -> CameraComponent:
        """
        Create Color camera
        """
        return CameraComponent(
            source, name, out
        )

    def create_nn(

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
                version = dai.OpenVINO.VERSION_2020_4, # 
                deviceInfo = getDeviceInfo(device),
                usb2Mode = usb2
                )
        else:
            return dai.Device(
                version = dai.OpenVINO.VERSION_2020_4, # 
                deviceInfo = getDeviceInfo(device),
                maxUsbSpeed = dai.UsbSpeed.SUPER_PLUS
                )

    @property
    def pipeline(self):
        return self.pipeline
        
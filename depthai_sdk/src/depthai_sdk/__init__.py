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
    # User should be able to access these:
    pipeline: dai.Pipeline
    devices: List[dai.Device] = []
    args = None # User defined arguments
    replay: Replay = None

    availableStreams = dict() # If recording is set, get available streams. If not, query device's cameras

    # TODO: 
    # - available streams; query cameras, or Replay.getStreams(). Pass these to camera component

    def __init__(self, 
        device: Optional[str] = None, # MxId / IP / USB port / "ALL"
        usb2: Optional[bool] = None, # Auto by default
        recording: Optional[str] = None,
        args: bool = True
        ) -> None:
        """
        Args:
            device (str, optional): OAK device we want to connect to
            usb2 (bool, optional): Force USB2 mode
            recording (str, optional): Use depthai-recording - either local path, or from depthai-recordings repo
            args (bool): Use user defined arguments when constructing the pipeline
        """

        self.pipeline = dai.Pipeline()
        self.devices.append(self._get_device(device, usb2))
        print(self.devices[0].getConnectedCameras())

        if args:
            am = ArgsManager()
            self.args = am.parseArgs()

        if recording is not None:
            self.replay = self._getReplay(recording)
            self.replay.initPipeline(self.pipeline)
            print(self.replay.getStreams())

    def _getReplay(self, path: str) -> Replay:
        """
        Either use local depthai-recording, or (TODO) download it from depthai-recordings
        """
        return Replay(path)

    def create_camera(self,
        source: Optional[str] = None,
        name: Optional[str] = None,
        out: bool = False,
        encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
        control: bool = False,
        ) -> CameraComponent:
        """
        Create Color camera
        """
        return CameraComponent(
            pipeline=self.pipeline,
            source=self.replay if self.replay is not None else source,
            name=name,
            out=out,
            encode=encode,
            control=control,
            args = self.args,
        )

    def create_nn(self,
        ) -> NNComponent:
        a = 5
        

    def _get_device(self,
        device: Optional[str] = None,
        usb2: Optional[bool] = None) -> dai.Device:
        """
        Connect to the OAK camera(s) and return dai.Device object
        """
        if device is not None and device.upper() == "ALL":
            # Connect to all available cameras
            raise NotImplementedError("TODO")
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
            self.pipeline.setXLinkChunkSize(xlinkChunk)
        if calib is not None:
            self.pipeline.setCalibrationData(calib)
        if tuningBlob is not None:
            self.pipeline.setCameraTuningBlobPath(tuningBlob)
    
    def __del__(self):
        for device in self.devices:
            device.close()

    @property
    def device(self) -> dai.Device:
        return self.devices[0]
        
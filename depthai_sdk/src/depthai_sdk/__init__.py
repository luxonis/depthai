from cProfile import label
from .fps import *
from .previews import *
from .utils import *
from .managers import *
from .record import *
from .replay import *
from .components import *

from typing import Optional
import depthai as dai
from pathlib import Path


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
    replay: Optional[Replay] = None

    _availableCameras = dict() # If recording is set, get available streams. If not, query device's cameras
    cameras = SimpleNamespace() # Already inited cameras (color/monos/stereo)


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

        if recording:
            self.replay = self._getReplay(recording)
            self.replay.initPipeline(self.pipeline)
            print('available streams from recording', self.replay.getStreams())

    def _getReplay(self, path: str) -> Replay:
        """
        Either use local depthai-recording, YT link, (TODO) mp4 url, or recording from depthai-recordings repo
        """
        if self._isUrl(path):
            if self._isYoutubeLink(path):
                from utils import downloadYTVideo
                # Overwrite source - so Replay class can use it
                path = str(downloadYTVideo(path))
            else:
                # TODO: download video/image(s) from the internet
                raise NotImplementedError("Only YouTube video download is currently supported!")

        p = Path(path)
        if not p.is_file:
            raise NotImplementedError("TODO: download from depthai-recordings repo")
            
        return Replay(path)

    
    def _isYoutubeLink(self, source: str) -> bool:
        return "youtube.com" in source

    def _isUrl(self, source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")

    def create_camera(self,
        source: str,
        name: Optional[str] = None,
        out: Union[None, bool, str] = None,
        encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
        control: bool = False,
        ) -> CameraComponent:
        """
        Create Color camera
        """
        return CameraComponent(
            pipeline=self.pipeline,
            source=source,
            name=name,
            out=out,
            encode=encode,
            control=control,
            replay=self.replay,
            args = self.args,
        )

    def create_nn(self,
        model: Union[str, Path], # str for SDK supported model or Path to custom model's json
        input: Union[CameraComponent, NNComponent, dai.Node.Output],
        out: Union[None, bool, str] = None,
        type: Optional[str] = None,
        name: Optional[str] = None, # name of the node
        tracker: bool = False, # Enable object tracker - only for Object detection models
        spatial: bool = False, # 
        ) -> NNComponent:
        return NNComponent(
            pipeline=self.pipeline,
            model=model,
            input=input,
            out=out,
            nnType=type,
            name=name,
            tracker=tracker,
            spatial=spatial,
            args=self.args
        )

    def create_stereo(self,
        resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
        fps: Optional[float] = None,
        name: Optional[str] = None,
        out: Optional[str] = None, # 'depth', 'disparity', both seperated by comma? TBD
        left: Union[None, dai.Node.Output, CameraComponent] = None, # Left mono camera
        right: Union[None, dai.Node.Output, CameraComponent] = None, # Right mono camera
        encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
        control: bool = False,
        ) -> StereoComponent:
        """
        Create Stereo camera component
        """
        return StereoComponent(
            pipeline=self.pipeline,
            resolution=resolution,
            fps=fps,
            name=name,
            out=out,
            left=left,
            right=right,
            encode=encode,
            control=control,
            replay=self.replay,
            args = self.args,
        )

    def _get_device(self,
        device: Optional[str] = None,
        usb2: Optional[bool] = None) -> dai.Device:
        """
        Connect to the OAK camera(s) and return dai.Device object
        """
        if device and device.upper() == "ALL":
            # Connect to all available cameras
            raise NotImplementedError("TODO")
        if usb2:
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
        if xlinkChunk:
            self.pipeline.setXLinkChunkSize(xlinkChunk)
        if calib:
            self.pipeline.setCalibrationData(calib)
        if tuningBlob:
            self.pipeline.setCameraTuningBlobPath(tuningBlob)
    
    def __del__(self):
        for device in self.devices:
            print("Closing OAK camera")
            device.close()
    
    def start(self) -> None:
        """
        Start the application. Configure XLink queues, upload the pipeline to the device(s)
        """
        if True:
            # Debug pipeline with pipeline graph tool
            folderPath = Path(os.path.abspath(sys.argv[0])).parent
            with open(folderPath / "pipeline.json", 'w') as f:
                f.write(json.dumps(self.pipeline.serializeToJson()['pipeline']))

        for device in self.devices:
            device.startPipeline(self.pipeline)

        # TODO: Go through each component, check if out is enabled
        for device in self.devices:
            if self.replay:
                self.replay.createQueues(device)



    def running(self) -> bool:
        """
        Check whether device is running. If we are using depthai-recording, send msgs to the device.
        """
        if self.replay:
            return self.replay.sendFrames()
        
        # TODO: check if device is closed
        return True


    @property
    def device(self) -> dai.Device:
        return self.devices[0]
        
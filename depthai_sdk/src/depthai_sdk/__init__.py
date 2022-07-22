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

    class OakDevice:
        device: dai.Device
        # str: Name (XLinkOut stream name, or replay stream)
        # Type: Component name, or Replay
        queues: Dict[str, Type] = {}
        messages: Dict[str, List[dai.Buffer]] = {} # List of messages

        # Dict of sequences, with Dict of stream names and their dai message(s)
        seqFrames: Dict[str, Dict[str, Any]] = {}

        @property
        def imageSensors(self) -> List[dai.CameraBoardSocket]:
            """
            Available imageSensors available on the camera
            """
            return self.device.getConnectedCameras()
        @property
        def info(self) -> dai.DeviceInfo: return self.device.getDeviceInfo()

        _xoutNames: List[str] = None
        @property
        def xoutNames(self) -> List[str]:
            if not self._xoutNames:
                self._xoutNames = []
                for qName, qType in self.queues.items():
                    if qType == Replay:
                        continue
                    self._xoutNames.append(qName)
            return self._xoutNames

        _replayNames: List[str] = None
        @property
        def replayNames(self) -> List[str]:
            if not self._replayNames:
                self._replayNames = []
                for qName, qType in self.queues.items():
                    if qType != Replay:
                        continue
                    self._replayNames.append(qName)
            return self._replayNames

    # User should be able to access these:
    pipeline: dai.Pipeline
    devices: List[OakDevice]
    args = None # User defined arguments
    replay: Optional[Replay] = None
    components: List[Component] = [] # List of components

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
        self._init_devices(device, usb2)

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

    def _comp(self, comp: Component) -> Component:
        self.components.append(comp)
        return comp

    def create_camera(self,
        source: str,
        resolution: Union[None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None,
        fps: Optional[float] = None,
        name: Optional[str] = None,
        out: Union[None, bool, str] = None,
        encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
        control: bool = False,
        ) -> CameraComponent:
        """
        Create Color camera
        """
        return self._comp(CameraComponent(
            pipeline=self.pipeline,
            source=source,
            resolution=resolution,
            fps=fps,
            name=name,
            out=out,
            encode=encode,
            control=control,
            replay=self.replay,
            args = self.args,
        ))

    def create_nn(self,
        model: Union[str, Path], # 
        input: Union[CameraComponent, NNComponent, dai.Node.Output],
        out: Union[None, bool, str] = None,
        type: Optional[str] = None,
        name: Optional[str] = None, # name of the node
        tracker: bool = False, # Enable object tracker - only for Object detection models
        spatial: Union[None, bool, StereoComponent, dai.Node.Output] = None,
        ) -> NNComponent:
        """
        Create NN component.
        Args:
            model (str / Path): str for SDK supported model or Path to custom model's json
            input (Component / dai.Node.Output): Input to the model. If NNComponent (detector), it creates 2-stage NN
            out (str / bool): Stream results to the host
            type (str): Type of the network (yolo/mobilenet) for on-device NN result decoding
            tracker: Enable object tracker, if model is object detector (yolo/mobilenet)
            spatial: Calculate 3D spatial coordinates, if model is object detector (yolo/mobilenet) and depth stream is available
        """
        return self._comp(NNComponent(
            pipeline=self.pipeline,
            model=model,
            input=input,
            out=out,
            nnType=type,
            name=name,
            tracker=tracker,
            spatial=spatial,
            args=self.args
        ))

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
        return self._comp(StereoComponent(
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
        ))

    def _init_devices(self,
        device: Optional[str] = None,
        usb2: Optional[bool] = None) -> None:
        """
        Connect to the OAK camera(s) and return dai.Device object
        """
        self.devices = []
        if device and device.upper() == "ALL":
            # Connect to all available cameras
            raise NotImplementedError("TODO")

        obj = self.OakDevice()
        if usb2:
            obj.device = dai.Device(
                version = dai.OpenVINO.VERSION_2021_4,
                deviceInfo = getDeviceInfo(device),
                usb2Mode = usb2
            )
        else:
            obj.device = dai.Device(
                version = dai.OpenVINO.VERSION_2021_4,
                deviceInfo = getDeviceInfo(device),
                maxUsbSpeed = dai.UsbSpeed.SUPER_PLUS
                )
        self.devices.append(obj)

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

    def _get_new_msgs(self, daiDevice: OakDevice) -> None:
        """
        Get messages from the device. Blocking behaviour. It will insert messages directly into daiDevice.messages
        List[dai.Messages]
        """
        # First check whether there are existing messages in queues
        flag = False
        for qName in daiDevice.xoutNames:
            if daiDevice.device.getOutputQueue(qName).has():
                flag = True
                daiDevice.messages[qName].extend(daiDevice.device.getOutputQueue(qName).getAll())
                # print(time.time(),"Getting msgs from local", qName)
        if flag: return

        qName = daiDevice.device.getQueueEvent(daiDevice.xoutNames)
        daiDevice.messages[qName].extend(daiDevice.device.getOutputQueue(qName).getAll())
        # print(time.time(),"Waited for msgs from", qName)

    def get_msgs(self, daiDevice: Optional[OakDevice] = None) -> Dict[str, dai.Buffer]:
        """
        Get all messages (not synced). Blocking until a new frame arrives.
        """
        if daiDevice is None: # Only 1 device
            if 1 < len(self.devices):
                raise ValueError("You have multiple OAK devices connected! Pass the OakDevice object to specify \
                from which OAK device you want to get a new message.")

        daiDevice = self.oakDevice

        # Get latest msgs
        self._wait_for_new_msg(daiDevice)

        msgs: Dict[str, dai.Buffer] = dict()
        for name, msgList in daiDevice.messages.items():
            msgs[name] = msgList[-1]
            # Clear all old msgs
            msgList = msgList[len(msgList) -1:]
            print('Msg list clear', msgList)

        
        return msgs

    def _wait_for_new_msg(self, daiDevice: Optional[OakDevice] = None) -> None:
        """
        Waits to receive a new message from the device. Function will block until at least one message from
        each queue has been received.
        """
        while True:
            self._get_new_msgs(daiDevice)

            # If Replay module, get frame
            for replayName in daiDevice.replayNames:
                imgFrame = self.replay.imgFrames[replayName]
                if len(daiDevice.messages[replayName]) == 0 or daiDevice.messages[replayName][-1].getSequenceNum() != imgFrame.getSequenceNum():
                    daiDevice.messages[replayName].append(imgFrame)

            # Check if all frames available were received
            if len(daiDevice.messages) == len(daiDevice.queues):
                return

    def get_synced_msgs(self, daiDevice: Optional[OakDevice] = None) -> Dict[str, dai.Buffer]:
        raise NotImplementedError()
        
    
    def __del__(self):
        self.close()

    def close(self):
        for oak in self.devices:
            print("Closing OAK camera")
            oak.device.close()
    
    def start(self) -> None:
        """
        Start the application. Configure XLink queues, upload the pipeline to the device(s)
        """
        if True:
            # Debug pipeline with pipeline graph tool
            folderPath = Path(os.path.abspath(sys.argv[0])).parent
            with open(folderPath / "pipeline.json", 'w') as f:
                f.write(json.dumps(self.pipeline.serializeToJson()['pipeline']))

        for oakDev in self.devices:
            oakDev.device.startPipeline(self.pipeline)

        if self.replay:
            for oakDevice in self.devices:
                self.replay.createQueues(oakDevice.device)

        # Go through each component, check if out is enabled
        for component in self.components:
            for qName, type in component.xouts.items():
                for dev in self.devices:
                    dev.messages[qName] = []
                    dev.queues[qName] = type

    def running(self) -> bool:
        """
        Check whether device is running. If we are using depthai-recording, send msgs to the device.
        """
        if self.replay:
            return self.replay.sendFrames()
        
        # TODO: check if device is closed
        return True


    @property
    def oakDevice(self) -> OakDevice: return self.devices[0]
    @property
    def device(self) -> dai.Device: return self.devices[0].device
        
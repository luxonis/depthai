from .fps import *
from .previews import *
from .utils import *
from .managers import *
from .record import *
from .replay import *
from .components import *
from .classes import *
from .visualizing import *

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
    devices: List[OakDevice]
    args = None  # User defined arguments
    replay: Optional[Replay] = None
    components: List[Component] = []  # List of components

    # TODO: 
    # - available streams; query cameras, or Replay.getStreams(). Pass these to camera component

    def __init__(self,
                 device: Optional[str] = None,  # MxId / IP / USB port / "ALL"
                 usb2: Optional[bool] = None,  # Auto by default
                 recording: Optional[str] = None,
                 openvinoVersion: Union[None, str, dai.OpenVINO.Version] = None,
                 args: bool = True
                 ) -> None:
        """
        Args:
            device (str, optional): OAK device we want to connect to
            usb2 (bool, optional): Force USB2 mode
            recording (str, optional): Use depthai-recording - either local path, or from depthai-recordings repo
            openvinoVersion: Specify OpenVINO version
            args (bool): Use user defined arguments when constructing the pipeline
        """

        self.pipeline = dai.Pipeline()
        self._init_devices(device, usb2, openvinoVersion)

        if args:
            am = ArgsManager()
            self.args = am.parseArgs()

        if recording:
            self.replay = Replay(recording)
            self.replay.initPipeline(self.pipeline)
            print('available streams from recording', self.replay.getStreams())

    def _comp(self, comp: Component) -> Component:
        self.components.append(comp)
        return comp

    def create_camera(self,
                      source: str,
                      resolution: Union[
                          None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None,
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
            args=self.args,
        ))

    def create_nn(self,
                  model: Union[str, Path],  #
                  input: Union[CameraComponent, NNComponent, dai.Node.Output],
                  out: Union[None, bool, str] = None,
                  type: Optional[str] = None,
                  name: Optional[str] = None,  # name of the node
                  tracker: bool = False,  # Enable object tracker - only for Object detection models
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
                      out: Optional[str] = None,  # 'depth', 'disparity', both seperated by comma? TBD
                      left: Union[None, dai.Node.Output, CameraComponent] = None,  # Left mono camera
                      right: Union[None, dai.Node.Output, CameraComponent] = None,  # Right mono camera
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
            args=self.args,
        ))

    def create_visualizer(self, components: List[Component], scale: Union[None, float, Tuple[int, int]], fps=False):
        vis = Visualizer(components, scale, fps)
        self.callback(components, vis.newMsgs)
        return vis

    def _init_devices(self,
                      device: Optional[str] = None,
                      usb2: Optional[bool] = None,
                      openvinoVersion: Union[None, str, dai.OpenVINO.Version] = None,
                      ) -> None:
        """
        Connect to the OAK camera(s) and return dai.Device object
        """
        self.devices = []
        if device and device.upper() == "ALL":
            # Connect to all available cameras
            raise NotImplementedError("TODO")

        obj = OakDevice()
        if usb2:
            obj.device = dai.Device(
                version=parseOpenVinoVersion(openvinoVersion),
                deviceInfo=getDeviceInfo(device),
                usb2Mode=usb2
            )
        else:
            obj.device = dai.Device(
                version=parseOpenVinoVersion(openvinoVersion),
                deviceInfo=getDeviceInfo(device),
                maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS
            )
        self.devices.append(obj)

    def configPipeline(self,
                       xlinkChunk: Optional[int] = None,
                       calib: Optional[dai.CalibrationHandler] = None,
                       tuningBlob: Optional[str] = None,
                       openvinoVersion: Union[None, str, dai.OpenVINO.Version] = None
                       ) -> None:
        configPipeline(self.pipeline, xlinkChunk, calib, tuningBlob, openvinoVersion)

    def __enter__(self):
        print('enter')
        return self

    def __exit__(self, exc_type, exc_value, tb):
        print('EXIT')
        for oak in self.devices:
            print("Closing OAK camera")
            oak.device.close()
        if self.replay:
            print("Closing replay")
            self.replay.close()

    def start(self, blocking=False) -> None:
        """
        Start the application. Configure XLink queues, upload the pipeline to the device(s)
        """
        if False:
            # Debug pipeline with pipeline graph tool
            folderPath = Path(os.path.abspath(sys.argv[0])).parent
            with open(folderPath / "pipeline.json", 'w') as f:
                f.write(json.dumps(self.pipeline.serializeToJson()['pipeline']))

        # Go through each component, check if out is enabled
        for component in self.components:
            for qName, (compType, daiMsgType) in component.xouts.items():
                for dev in self.devices:
                    dev.queues[qName] = compType

        for oakDev in self.devices:
            oakDev.device.startPipeline(self.pipeline)
            oakDev.initCallbacks()

        if self.replay:
            self.replay.createQueues(self.device)
            self.replay.start(self._replay_callback)

        # Check if callbacks (sync/non-sync are set)
        if blocking:
            # Constant loop: get messages, call callbacks
            while True:
                if not self.poll(): break

    def _replay_callback(self, name, msg):
        """
        Called from Replay module on each new frame sent to the device.
        """
        for oakDevice in self.devices:
            if name in oakDevice.replayNames:
                oakDevice.newMsg(name, msg)

    def poll(self) -> bool:
        """
        Poll events; cv2.waitKey, send controls to OAK (if controls are enabled), update, check syncs.
        True if successful.
        """
        key = cv2.waitKey(1)
        if key == ord('q'): return False

        for oakDevice in self.devices:
            oakDevice.checkSync()

        return True

    def running(self) -> bool:
        # TODO: check if device is closed
        return True

    def synchronize(self, components: List[Component], callback: Callable):
        raise NotImplementedError()

    def callback(self, components: List[Component], function: Callable):
        streams = []
        for comp in components:
            streams.extend([name for name, _ in comp.xouts.items()])

        for oakDevice in self.devices:
            oakDevice.sync.append(NoSync(function, streams))

    @property
    def oakDevice(self) -> OakDevice:
        return self.devices[0]

    @property
    def device(self) -> dai.Device:
        return self.devices[0].device

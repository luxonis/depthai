from .fps import *
from .previews import *
from .syncing import *
from .utils import *
from .managers import *
from .record import *
from .replay import *
from .components import *
from .classes import *
from .visualizing import *
from .oak_device import OakDevice

from typing import Optional
import depthai as dai
from pathlib import Path


class OakCamera:
    """
    TODO: Write useful comments for users

    Camera is the main abstraction class for the OAK cameras. It abstracts pipeline building, different camera permutations,
    AI model handling, visualization, user arguments, syncing, etc.

    This abstraction layer will internally use SDK Components.
    """

    # User should be able to access these:
    pipeline: dai.Pipeline = None
    oak: OakDevice = OakDevice()  # Init this object by default
    args = None  # User defined arguments
    replay: Optional[Replay] = None
    components: List[Component] = []  # List of components

    usb2: bool = False  # Whether to force USB2 mode
    deviceName: str = None  # MxId / IP / USB port

    visualizers: List[Visualizer] = []
    syncs: List[Sync] = []

    # TODO: 
    # - available streams; query cameras, or Replay.getStreams(). Pass these to camera component

    def __init__(self,
                 device: Optional[str] = None,  # MxId / IP / USB port
                 usb2: Optional[bool] = None,  # Auto by default
                 recording: Optional[str] = None,
                 args: Union[None, bool, Dict] = None
                 ) -> None:
        """
        Args:
            device (str, optional): OAK device we want to connect to
            usb2 (bool, optional): Force USB2 mode
            recording (str, optional): Use depthai-recording - either local path, or from depthai-recordings repo
            args (None, bool, Dict): Use user defined arguments when constructing the pipeline
        """
        self.deviceName = device
        self.usb2 = usb2

        if args is not None:
            if isinstance(args, bool) and args:
                am = ArgsManager()
                self.args = am.parseArgs()
            else:  # Already parsed
                self.args = args

        if recording:
            self.replay = Replay(recording)
            print('available streams from recording', self.replay.getStreams())


    def _comp(self, comp: Component) -> Union[CameraComponent, NNComponent, StereoComponent]:
        self.components.append(comp)
        return comp

    def create_camera(self,
                      source: str,
                      resolution: Union[
                          None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None,
                      fps: Optional[float] = None,
                      out: Union[None, bool, str] = None,
                      encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                      control: bool = False,
                      ) -> CameraComponent:
        """
        Create Color camera
        """
        return self._comp(CameraComponent(
            source=source,
            resolution=resolution,
            fps=fps,
            out=out,
            encode=encode,
            control=control,
            replay=self.replay,
            args=self.args,
        ))

    def create_nn(self,
                  model: Union[str, Path],
                  input: Union[CameraComponent, NNComponent, dai.Node.Output],
                  out: Union[None, bool, str] = None,
                  type: Optional[str] = None,
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
            model=model,
            input=input,
            out=out,
            nnType=type,
            tracker=tracker,
            spatial=spatial,
            args=self.args
        ))

    def create_stereo(self,
                      resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
                      fps: Optional[float] = None,
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
            resolution=resolution,
            fps=fps,
            out=out,
            left=left,
            right=right,
            encode=encode,
            control=control,
            replay=self.replay,
            args=self.args,
        ))

    def _init_device(self) -> None:
        """
        Connect to the OAK camera
        """
        if self.deviceName:
            deviceInfo = dai.DeviceInfo(self.deviceName)
        else:
            (found, deviceInfo) = dai.DeviceBootloader.getFirstAvailableDevice()
            if not found:
                raise Exception("No OAK device found to connect to!")

        version = self.pipeline.getOpenVINOVersion()
        if self.usb2:
            self.oak.device = dai.Device(
                version=version,
                deviceInfo=deviceInfo,
                usb2Mode=self.usb2
            )
        else:
            self.oak.device = dai.Device(
                version=version,
                deviceInfo=deviceInfo,
                maxUsbSpeed=dai.UsbSpeed.SUPER
            )

    def config_pipeline(self,
                        xlinkChunk: Optional[int] = None,
                        calib: Optional[dai.CalibrationHandler] = None,
                        tuningBlob: Optional[str] = None,
                        openvinoVersion: Union[None, str, dai.OpenVINO.Version] = None
                        ) -> None:
        configPipeline(self.pipeline, xlinkChunk, calib, tuningBlob, openvinoVersion)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        print("Closing OAK camera")
        self.oak.device.close()
        print('Device closed')

        if self.replay:
            print("Closing replay")
            self.replay.close()

    def start(self, blocking=False) -> None:
        """
        Start the application
        """
        if self.pipeline is None:
            self.build() # Build the pipeline

        # Set up Syncing classes and visualizers
        for vis in self.visualizers:
            vis.setup()
        for sync in self.syncs:
            sync.setup()
            self.oak.sync.append(sync.base)

        self.oak.device.startPipeline(self.pipeline)
        self.oak.initCallbacks()

        if self.replay:
            self.replay.createQueues(self.oak.device)
            # Called from Replay module on each new frame sent to the device.
            self.replay.start(self.oak.newMsg)

        # Check if callbacks (sync/non-sync are set)
        if blocking:
            # Constant loop: get messages, call callbacks
            while self.running():
                time.sleep(0.0005)
                if not self.poll():
                    break

    def poll(self) -> bool:
        """
        Poll events; cv2.waitKey, send controls to OAK (if controls are enabled), update, check syncs.
        True if successful.
        """
        key = cv2.waitKey(1)
        if key == ord('q'):
            return False

        self.oak.checkSync()

        return True

    def running(self) -> bool:
        return True

    def build(self) -> None:
        """
        Connect to the device and build the pipeline based on previously provided configuration. This is called
        by start() function. Configure XLink queues, upload the pipeline to the device.
        """
        self.pipeline = dai.Pipeline()
        if self.replay:
            self.replay.initPipeline(self.pipeline)

        # First go through each components to check whether any is forcing an OpenVINO version
        for c in self.components:
            ov = c._forced_openvino_version()
            if ov:
                if self.pipeline.getRequiredOpenVINOVersion() and self.pipeline.getRequiredOpenVINOVersion() != ov:
                    raise Exception(
                        'Two components forced two different OpenVINO version! Please make sure that all your models are compiled using the same OpenVINO version.')
                self.pipeline.setOpenVINOVersion(ov)

        # Connect to the OAK camera
        self._init_device()

        # Go through each component
        for component in self.components:
            # Update the component now that we can query device info
            component._update_device_info(self.pipeline, self.oak.device, self.pipeline.getOpenVINOVersion())

            # check if out is enabled
            for qName, (compType, daiMsgType) in component.xouts.items():
                self.oak.queues[qName] = compType


    def show_graph(self) -> None:
        """
        Show pipeline graph, useful for debugging.
        @return:
        """
        if self.pipeline is None:
            self.build() # Build the pipeline

        p = PipelineGraph(self.pipeline.serializeToJson()['pipeline'])


    def visualize(self, components: List[Component],
                  scale: Union[None, float, Tuple[int, int]] = None,
                  fps=False,
                  callback: Callable=None) -> Visualizer:
        handlers = None
        if fps:
            self.oak.enable_fps(True)
            handlers = self.oak.fpsHandlers

        vis = Visualizer(components, scale, handlers, callback)
        self.visualizers.append(vis)
        self.synchronize(components, vis.new_msgs)
        return vis

    def synchronize(self, components: List[Component], callback: Callable) -> Sync:
        sync = Sync(components, callback)
        self.syncs.append(sync)
        return sync

    @property
    def device(self) -> dai.Device:
        return self.oak.device

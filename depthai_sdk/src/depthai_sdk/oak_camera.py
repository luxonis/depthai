import functools
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple

import cv2
import depthai as dai

from .visualize import Visualizer
from .args_parser import ArgsParser
from .classes.output_config import BaseConfig, RecordConfig, OutputConfig
from .components.camera_component import CameraComponent
from .components.component import Component
from .components.imu_component import IMUComponent
from .components.nn_component import NNComponent
from .components.parser import parse_usb_speed
from .components.pipeline_graph import PipelineGraph
from .components.stereo_component import StereoComponent
from .oak_device import OakDevice
from .record import RecordType, Record
from .replay import Replay
from .utils import configPipeline


class UsbWarning(UserWarning):
    pass


def _add_to_components(func) -> Callable:
    """
    Decorator to add created component to the components list.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Component:
        comp = func(self, *args, **kwargs)
        self._components.append(comp)
        return comp

    return wrapper


class OakCamera:
    """
    OakCamera improves ease of use when developing apps for OAK devices.

    It abstracts DepthAI API pipeline building, different camera permutations, stream recording/replaying, it adds
    debugging features, does AI model handling, message syncing & visualization, and much more.

    It was designed with interoperability with depthai API in mind.
    """

    # User should be able to access these:
    _pipeline: dai.Pipeline = None
    _oak: OakDevice  # Init this object by default
    _args: Dict[str, Any] = None  # User defined arguments
    replay: Optional[Replay] = None
    _components: List[Component] = []  # List of components

    _usb_speed: Optional[dai.UsbSpeed] = None
    _device_name: str = None  # MxId / IP / USB port

    _out_templates: List[BaseConfig] = []

    def __init__(self,
                 device: Optional[str] = None,  # MxId / IP / USB port
                 usbSpeed: Union[None, str, dai.UsbSpeed] = None,  # Auto by default
                 recording: Optional[str] = None,
                 args: Union[bool, Dict] = True
                 ):
        """
        Initializes OakCamera

        Args:
            device (str, optional): OAK device we want to connect to
            usb2 (bool, optional): Force USB2 mode
            recording (str, optional): Use depthai-recording - either local path, or from depthai-recordings repo
            args (None, bool, Dict): Use user defined arguments when constructing the pipeline
        """
        self._device_name = device
        self._usb_speed = parse_usb_speed(usbSpeed)
        self._oak = OakDevice()
        self._pipeline = dai.Pipeline()
        self._pipeline_built = False

        if args:
            if isinstance(args, bool):
                if args:
                    self._args = ArgsParser.parseArgs()
                    # Set up the OakCamera
                    if self._args.get('recording', None):
                        recording = self._args.get('recording', None)
                    if self._args.get('deviceId', None):
                        self._device_name = self._args.get('deviceId', None)
                    if self._args.get('usbSpeed', None):
                        self._usb_speed = parse_usb_speed(self._args.get('usbSpeed', None))

                # else False - we don't want to parse user arguments
            else:  # Already parsed
                self._args = args

        if recording:
            self.replay = Replay(recording)
            print('Available streams from recording:', self.replay.getStreams())

    def _comp(self, comp: Component) -> Union[CameraComponent, NNComponent, StereoComponent]:
        self._components.append(comp)
        return comp

    @_add_to_components
    def create_camera(self,
                      source: str,
                      resolution: Union[
                          None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None,
                      fps: Optional[float] = None,
                      encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                      ) -> CameraComponent:
        """
        Creates Camera component. This abstracts ColorCamera/MonoCamera nodes and supports mocking the camera when
        recording is passed during OakCamera initialization. Mocking the camera will send frames from the host to the
        OAK device (via XLinkIn node).

        Args:
            source (str): Either 'color', 'left' or 'right' (these 2 will create MonoCamera nodes)
            resolution (str/SensorResolution): Sensor resolution of the camera.
            fps (float): Sensor FPS
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via cameraComponent.out_encoded). If True, it will use MJPEG
        """
        return CameraComponent(
            self._pipeline,
            source=source,
            resolution=resolution,
            fps=fps,
            encode=encode,
            replay=self.replay,
            args=self._args,
        )

    @_add_to_components
    def create_nn(self,
                  model: Union[str, Path],
                  input: Union[CameraComponent, NNComponent],
                  type: Optional[str] = None,
                  tracker: bool = False,  # Enable object tracker - only for Object detection models
                  spatial: Union[None, bool, StereoComponent] = None,
                  ) -> NNComponent:
        """
        Creates Neural Network component.

        Args:
            model (str / Path): str for SDK supported model or Path to custom model's json/blob
            input (CameraComponent/NNComponent): Input to the model. If NNComponent (detector), it creates 2-stage NN
            type (str): Type of the network (yolo/mobilenet) for on-device NN result decoding (only needed if blob path was specified)
            tracker: Enable object tracker, if model is object detector (yolo/mobilenet)
            spatial: Calculate 3D spatial coordinates, if model is object detector (yolo/mobilenet) and depth stream is available
        """
        return NNComponent(
            self._pipeline,
            model=model,
            input=input,
            nnType=type,
            tracker=tracker,
            spatial=spatial,
            replay=self.replay,
            args=self._args
        )

    @_add_to_components
    def create_stereo(self,
                      resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
                      fps: Optional[float] = None,
                      left: Union[None, dai.Node.Output, CameraComponent] = None,  # Left mono camera
                      right: Union[None, dai.Node.Output, CameraComponent] = None,  # Right mono camera
                      ) -> StereoComponent:
        """
        Create Stereo camera component. If left/right cameras/component aren't specified they will get created internally.

        Args:
            resolution (str/SensorResolution): If monochrome cameras aren't already passed, create them and set specified resolution
            fps (float): If monochrome cameras aren't already passed, create them and set specified FPS
            left (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
            right (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
        """
        return StereoComponent(
            self._pipeline,
            resolution=resolution,
            fps=fps,
            left=left,
            right=right,
            replay=self.replay,
            args=self._args,
        )

    @_add_to_components
    def create_imu(self) -> IMUComponent:
        """
        Create IMU component
        """
        return IMUComponent(
            pipeline=self._pipeline
        )

    def _init_device(self) -> None:
        """
        Connect to the OAK camera
        """
        if self._device_name:
            deviceInfo = dai.DeviceInfo(self._device_name)
        else:
            (found, deviceInfo) = dai.Device.getFirstAvailableDevice()
            if not found:
                raise Exception("No OAK device found to connect to!")

        version = self._pipeline.getOpenVINOVersion()

        if self._usb_speed == dai.UsbSpeed.SUPER:
            self._oak.device = dai.Device(
                version=version,
                deviceInfo=deviceInfo,
                usb2Mode=True
            )
        else:
            self._oak.device = dai.Device(
                version=version,
                deviceInfo=deviceInfo,
                maxUsbSpeed=dai.UsbSpeed.SUPER if self._usb_speed is None else self._usb_speed
            )

        # TODO test with usb3 (SUPER speed)
        if self._usb_speed != dai.UsbSpeed.HIGH and self._oak.device.getUsbSpeed() == dai.UsbSpeed.HIGH:
            warnings.warn("Device connected in USB2 mode! This might cause some issues. "
                          "In such case, please try using a (different) USB3 cable, "
                          "or force USB2 mode 'with OakCamera(usbSpeed=depthai.UsbSpeed.HIGH)'", UsbWarning)

    def config_pipeline(self,
                        xlinkChunk: Optional[int] = None,
                        calib: Optional[dai.CalibrationHandler] = None,
                        tuningBlob: Optional[str] = None,
                        openvinoVersion: Union[None, str, dai.OpenVINO.Version] = None
                        ):
        """
        Configures DepthAI pipeline.
        @param xlinkChunk: Chunk size of XLink messages. 0 can result in lower latency
        @param calib: Calibration data to be uploaded to OAK
        @param tuningBlob: Camera tuning blob
        @param openvinoVersion: Force specific OpenVINO version
        """
        configPipeline(self._pipeline, xlinkChunk, calib, tuningBlob, openvinoVersion)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        print("Closing OAK camera")
        if self._oak.device is not None:
            self._oak.device.close()
        if self.replay:
            print("Closing replay")
            self.replay.close()

        for out in self._out_templates:
            if isinstance(out, RecordConfig):
                out.rec.close()

    def start(self, blocking=False):
        """
        Start the application - upload the pipeline to the OAK device.
        Args:
            blocking (bool):  Continuously loop and call oak.poll() until program exits
        """
        if not self._pipeline_built:
            self.build()  # Build the pipeline

        self._oak.device.startPipeline(self._pipeline)

        self._oak.initCallbacks(self._pipeline)

        for xout in self._oak.oak_out_streams:  # Start FPS counters
            xout.start_fps()

        if self.replay:
            self.replay.createQueues(self._oak.device)
            # Called from Replay module on each new frame sent to the device.
            self.replay.start(self._oak.newMsg)

        # Check if callbacks (sync/non-sync are set)
        if blocking:
            # Constant loop: get messages, call callbacks
            while True:
                time.sleep(0.001)
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

        # TODO: check if components have controls enabled and check whether key == `control`

        self._oak.checkSync()

        if self.replay:
            if self.replay._stop:
                return False

        return True  # TODO: check whether OAK is connected

    def build(self) -> dai.Pipeline:
        """
        Connect to the device and build the pipeline based on previously provided configuration. Configure XLink queues,
        upload the pipeline to the device. This function must only be called once!  build() is also called by start().
        Return:
            Built dai.Pipeline
        """
        if self._pipeline_built:
            raise Exception('Pipeline can be built only once!')

        self._pipeline_built = True
        if self.replay:
            self.replay.initPipeline(self._pipeline)

        # First go through each component to check whether any is forcing an OpenVINO version
        # TODO: check each component's SHAVE usage
        for c in self._components:
            ov = c._forced_openvino_version()
            if ov:
                if self._pipeline.getRequiredOpenVINOVersion() and self._pipeline.getRequiredOpenVINOVersion() != ov:
                    raise Exception(
                        'Two components forced two different OpenVINO version! Please make sure that all your models are compiled using the same OpenVINO version.')
                self._pipeline.setOpenVINOVersion(ov)

        if self._pipeline.getRequiredOpenVINOVersion() == None:
            # Force 2021.4 as it's better supported (blobconverter, compile tool) for now.
            self._pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

        # Connect to the OAK camera
        self._init_device()

        # Go through each component
        for component in self._components:
            # Update the component now that we can query device info
            component._update_device_info(self._pipeline, self._oak.device, self._pipeline.getOpenVINOVersion())

        # Create XLinkOuts based on visualizers/callbacks enabled

        # TODO: clean this up and potentially move elsewhere

        names = []
        for out in self._out_templates:
            xoutbase = out.setup(self._pipeline, self._oak.device, names)
            self._oak.oak_out_streams.append(xoutbase)

        # User-defined arguments
        if self._args:
            self.config_pipeline(
                xlinkChunk=self._args.get('xlinkChunkSize', None),
                tuningBlob=self._args.get('cameraTuning', None),
                openvinoVersion=self._args.get('openvinoVersion', None),
            )
            self.device.setIrLaserDotProjectorBrightness(self._args.get('irDotBrightness', None) or 0)
            self.device.setIrFloodLightBrightness(self._args.get('irFloodBrightness', None) or 0)

        return self._pipeline

    def record(self, outputs: Union[Callable, List[Callable]], path: str, type: RecordType = RecordType.VIDEO):
        """
        Record component outputs. This handles syncing multiple streams (eg. left, right, color, depth) and saving
        them to the computer in desired format (raw, mp4, mcap, bag..).
        Args:
            outputs (Component/Component output): Component output(s) to be recorded
            path: Folder path where to save these streams
        """
        if isinstance(outputs, Callable):
            outputs = [outputs]  # to list

        self._out_templates.append(RecordConfig(outputs, Record(Path(path).resolve(), type)))

    def show_graph(self):
        """
        Shows DepthAI Pipeline graph, which can be useful when debugging. Builds the pipeline (oak.build()).
        """
        if not self._pipeline_built:
            self.build()  # Build the pipeline

        PipelineGraph(self._pipeline.serializeToJson()['pipeline'])

    def _callback(self,
                  output: Union[List, Callable, Component],
                  callback: Callable,
                  visualizer: Visualizer = None,
                  record: Optional[str] = None):
        if isinstance(output, List):
            for element in output:
                self._callback(element, callback, visualizer, record)
            return

        if isinstance(output, Component):
            output = output.out.main

        self._out_templates.append(OutputConfig(output, callback, visualizer, record))

    def visualize(self, output: Union[List, Callable, Component],
                  record: Optional[str] = None,
                  callback: Callable = None):
        """
        Visualize component output(s). This handles output streaming (OAK->host), message syncing, and visualizing.
        Args:
            output (Component/Component output): Component output(s) to be visualized. If component is passed, SDK will visualize its default output (out())
            record: Path where to store the recording (visualization window name gets appended to that path), supported formats: mp4, avi
            callback: Instead of showing the frame, pass the Packet to the callback function, where it can be displayed
        """
        visualizer = Visualizer()
        self._callback(output, callback, visualizer, record)
        return visualizer

    def callback(self, output: Union[List, Callable, Component], callback: Callable):
        """
        Create a callback for the component output(s). This handles output streaming (OAK->Host) and message syncing.
        Args:
            output: Component output(s) to be visualized. If component is passed, SDK will visualize its default output (out())
            callback: Handler function to which the Packet will be sent
        """
        self._callback(output, callback)

    @property
    def device(self) -> dai.Device:
        """
        Returns dai.Device object. oak.built() has to be called before querying this property!
        """
        if not self._pipeline_built:
            raise Exception("OAK device wasn't booted yet, make sure to call oak.build() or oak.start()!")
        return self._oak.device

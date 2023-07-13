import copy
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

from depthai_sdk import CV2_HAS_GUI_SUPPORT
from depthai_sdk.visualize.visualizer import Visualizer

try:
    import cv2
except ImportError:
    cv2 = None

import depthai as dai

from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.args_parser import ArgsParser
from depthai_sdk.classes.output_config import BaseConfig, RecordConfig, OutputConfig, SyncConfig, RosStreamConfig, TriggerActionConfig
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.components.imu_component import IMUComponent
from depthai_sdk.components.nn_component import NNComponent
from depthai_sdk.components.parser import parse_usb_speed, parse_camera_socket
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.components.pointcloud_component import PointcloudComponent
from depthai_sdk.oak_device import OakDevice
from depthai_sdk.record import RecordType, Record
from depthai_sdk.replay import Replay
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger
from depthai_sdk.utils import configPipeline, report_crash_dump


class UsbWarning(UserWarning):
    pass


class OakCamera:
    """
    OakCamera improves ease of use when developing apps for OAK devices.

    It abstracts DepthAI API pipeline building, different camera permutations, stream recording/replaying, it adds
    debugging features, does AI model handling, message syncing & visualization, and much more.

    It was designed with interoperability with depthai API in mind.
    """

    def __init__(self,
                 device: Optional[str] = None,
                 usb_speed: Union[None, str, dai.UsbSpeed] = None,  # Auto by default
                 replay: Optional[str] = None,
                 rotation: int = 0,
                 config: dai.Device.Config = None,
                 args: Union[bool, Dict] = True
                 ):
        """
        Initializes OakCamera

        Args:
            device (str, optional): OAK device we want to connect to, either MxId, IP, or USB port
            usb_speed (str, optional): USB speed we want to use. Defaults to 'auto'.
            replay (str, optional): Replay a depthai-recording - either local path, or from depthai-recordings repo
            rotation (int, optional): Rotate the camera output by this amount of degrees, 0 by default, 90, 180, 270 are supported.
            args (None, bool, Dict): Use user defined arguments when constructing the pipeline
        """

        # User should be able to access these:
        self.replay: Optional[Replay] = None

        self.pipeline = dai.Pipeline()
        self._args: Optional[Dict[str, Any]] = None  # User defined arguments
        self._oak = OakDevice()

        if args:
            if isinstance(args, bool):
                self._args = ArgsParser.parseArgs()
                # Set up the OakCamera
                if self._args.get('recording', None):
                    replay = self._args.get('recording', None)
                if self._args.get('deviceId', None):
                    device = self._args.get('deviceId', None)
                if self._args.get('usbSpeed', None):
                    usb_speed = parse_usb_speed(self._args.get('usbSpeed', None))
            else:  # Already parsed
                self._args = args

        if config is None:
            config = dai.Device.Config()
            config.version = dai.OpenVINO.VERSION_UNIVERSAL
            max_speed = parse_usb_speed(usb_speed)
            if max_speed is not None:
                config.board.usb.maxSpeed = max_speed

        self._init_device(config, device)
        report_crash_dump(self.device)

        # Whether to stop running the OAK camera. Used by oak.running()
        self._stop = False
        self._built = False
        self._polling = []

        self._components: List[Component] = []  # List of components
        self._out_templates: List[BaseConfig] = []

        self._rotation = rotation

        if replay is not None:
            self.replay = Replay(replay)
            self.replay.initPipeline(self.pipeline)
            logging.info(f'Available streams from recording: {self.replay.getStreams()}')
    
    def camera(self,
               source: Union[str, dai.CameraBoardSocket],
               resolution: Optional[Union[
                   str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                   ]] = None,
               fps: Optional[float] = None,
               encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
               name: Optional[str] = None,
               ) -> CameraComponent:
        """
        Creates Camera component. This abstracts ColorCamera/MonoCamera nodes and supports mocking the camera when
        recording is passed during OakCamera initialization. Mocking the camera will send frames from the host to the
        OAK device (via XLinkIn node).

        Args:
            source (str / dai.CameraBoardSocket): Camera source
            resolution (str/SensorResolution): Sensor resolution of the camera.
            fps (float): Sensor FPS
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via cameraComponent.out_encoded). If True, it will use MJPEG
            name (str): Name used to identify the X-out stream. This name will also be associated with the frame in the callback function.
        """
        socket = source
        if isinstance(source, str):
            socket = parse_camera_socket(source.split(",")[0])
        for comp in self._components:
            if isinstance(comp, CameraComponent) and comp.node.getBoardSocket() == socket:
                return comp

        comp = CameraComponent(self._oak.device,
                               self.pipeline,
                               source=source,
                               resolution=resolution,
                               fps=fps,
                               encode=encode,
                               rotation=self._rotation,
                               replay=self.replay,
                               name=name,
                               args=self._args)
        self._components.append(comp)
        return comp

    def create_camera(self,
                      source: Union[str, dai.CameraBoardSocket],
                      resolution: Optional[Union[
                          str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                      ]] = None,
                      fps: Optional[float] = None,
                      encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                      name: Optional[str] = None,
                      ) -> CameraComponent:
        """
        Deprecated, use camera() instead.

        Creates Camera component. This abstracts ColorCamera/MonoCamera nodes and supports mocking the camera when
        recording is passed during OakCamera initialization. Mocking the camera will send frames from the host to the
        OAK device (via XLinkIn node).

        Args:
            source (str / dai.CameraBoardSocket): Camera source
            resolution (str/SensorResolution): Sensor resolution of the camera.
            fps (float): Sensor FPS
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via cameraComponent.out_encoded). If True, it will use MJPEG
            name (str): Name used to identify the X-out stream. This name will also be associated with the frame in the callback function.
        """
        return self.camera(source, resolution, fps, encode, name)

    def all_cameras(self,
                    resolution: Optional[Union[
                        str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                        ]] = None,
                    fps: Optional[float] = None,
                    encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                    ) -> List[CameraComponent]:
        """
        Creates Camera component for each camera sensors on the OAK camera.

        Args:
            resolution (str/SensorResolution): Sensor resolution of the camera.
            fps (float): Sensor FPS
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via cameraComponent.out_encoded). If True, it will use MJPEG
        """
        components: List[CameraComponent] = []
        # Loop over all available camera sensors
        for cam_sensor in self._oak.device.getConnectedCameraFeatures():
            comp = CameraComponent(self._oak.device,
                                   self.pipeline,
                                   source=cam_sensor.socket,
                                   resolution=resolution,
                                   fps=fps,
                                   encode=encode,
                                   rotation=self._rotation,
                                   replay=self.replay,
                                   name=None,
                                   args=self._args)
            components.append(comp)

        self._components.extend(components)
        return components

    def create_all_cameras(self,
                           resolution: Optional[Union[
                               str, dai.ColorCameraProperties.SensorResolution, 
                               dai.MonoCameraProperties.SensorResolution
                               ]] = None,
                           fps: Optional[float] = None,
                           encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                           ) -> List[CameraComponent]:
        """
        Deprecated, use all_cameras() instead.

        Creates Camera component for each camera sensors on the OAK camera.

        Args:
            resolution (str/SensorResolution): Sensor resolution of the camera.
            fps (float): Sensor FPS
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via cameraComponent.out_encoded). If True, it will use MJPEG
        """
        return self.all_cameras(resolution, fps, encode)

    def create_nn(self,
                  model: Union[str, Dict, Path],
                  input: Union[CameraComponent, NNComponent],
                  nn_type: Optional[str] = None,
                  tracker: bool = False,  # Enable object tracker - only for Object detection models
                  spatial: Union[None, bool, StereoComponent] = None,
                  decode_fn: Optional[Callable] = None,
                  name: Optional[str] = None
                  ) -> NNComponent:
        """
        Creates Neural Network component.

        Args:
            model (str / Path): str for SDK supported model or Path to custom model's json/blob
            input (CameraComponent/NNComponent): Input to the model. If NNComponent (detector), it creates 2-stage NN
            nn_type (str): Type of the network (yolo/mobilenet) for on-device NN result decoding (only needed if blob path was specified)
            tracker: Enable object tracker, if model is object detector (yolo/mobilenet)
            spatial: Calculate 3D spatial coordinates, if model is object detector (yolo/mobilenet) and depth stream is available
            decode_fn: Custom decoding function for the model's output
            name (str): Name used to identify the X-out stream. This name will also be associated with the frame in the callback function.
        """
        comp = NNComponent(self._oak.device,
                           self.pipeline,
                           model=model,
                           input=input,
                           nn_type=nn_type,
                           tracker=tracker,
                           spatial=spatial,
                           decode_fn=decode_fn,
                           replay=self.replay,
                           args=self._args,
                           name=name)
        self._components.append(comp)
        return comp

    def stereo(self,
               resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
               fps: Optional[float] = None,
               left: Union[None, dai.Node.Output, CameraComponent] = None,  # Left mono camera
               right: Union[None, dai.Node.Output, CameraComponent] = None,  # Right mono camera
               name: Optional[str] = None,
               encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None
               ) -> StereoComponent:
        """
        Create Stereo camera component. If left/right cameras/component aren't specified they will get created internally.

        Args:
            resolution (str/SensorResolution): If monochrome cameras aren't already passed, create them and set specified resolution
            fps (float): If monochrome cameras aren't already passed, create them and set specified FPS
            left (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
            right (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
            name (str): Name used to identify the X-out stream. This name will also be associated with the frame in the callback function.
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via StereoComponent.out.encoded). If True, it will use h264 codec.
        """
        if left is None:
            left = self.camera(source=dai.CameraBoardSocket.LEFT, resolution=resolution, fps=fps)
        if right is None:
            right = self.camera(source=dai.CameraBoardSocket.RIGHT, resolution=resolution, fps=fps)

        comp = StereoComponent(self._oak.device,
                               self.pipeline,
                               resolution=resolution,
                               fps=fps,
                               left=left,
                               right=right,
                               replay=self.replay,
                               args=self._args,
                               name=name,
                               encode=encode)
        self._components.append(comp)
        return comp

    def create_stereo(self,
                      resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
                      fps: Optional[float] = None,
                      left: Union[None, dai.Node.Output, CameraComponent] = None,  # Left mono camera
                      right: Union[None, dai.Node.Output, CameraComponent] = None,  # Right mono camera
                      name: Optional[str] = None,
                      encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None
                      ) -> StereoComponent:
        """
        Deprecated, use stereo() instead.

        Create Stereo camera component. If left/right cameras/component aren't specified they will get created internally.

        Args:
            resolution (str/SensorResolution): If monochrome cameras aren't already passed, create them and set specified resolution
            fps (float): If monochrome cameras aren't already passed, create them and set specified FPS
            left (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
            right (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
            name (str): Name used to identify the X-out stream. This name will also be associated with the frame in the callback function.
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via StereoComponent.out.encoded). If True, it will use h264 codec.
        """
        return self.stereo(resolution, fps, left, right, name, encode)

    def create_imu(self) -> IMUComponent:
        """
        Create IMU component
        """
        comp = IMUComponent(self._oak.device, self.pipeline)
        self._components.append(comp)
        return comp

    def create_pointcloud(self,
                          stereo: Union[None, StereoComponent, dai.node.StereoDepth, dai.Node.Output] = None,
                          colorize: Union[None, CameraComponent, dai.node.MonoCamera, dai.node.ColorCamera, dai.Node.Output, bool] = None,
                          name: Optional[str] = None,
                          ) -> PointcloudComponent:

        if colorize is None:
            for component in self._components:
                if isinstance(component, CameraComponent):
                    if component.is_color():
                        colorize = component
                        break
                    else:
                        # ColorCamera has priority
                        colorize = component

        comp = PointcloudComponent(
            self._oak.device,
            self.pipeline,
            stereo=stereo,
            colorize=colorize,
            replay=self.replay,
            args=self._args,
            name=name
        )
        self._components.append(comp)
        return comp

    def _init_device(self,
                     config: dai.Device.Config,
                     device_str: Optional[str] = None,
                     ) -> None:

        """
        Connect to the OAK camera
        """
        if device_str is not None:
            device_info = dai.DeviceInfo(device_str)
        else:
            (found, device_info) = dai.Device.getFirstAvailableDevice()
            if not found:
                raise Exception("No OAK device found to connect to!")

        self._oak.device = dai.Device(
            config=config,
            deviceInfo=device_info,
        )

        # TODO test with usb3 (SUPER speed)
        if config.board.usb.maxSpeed != dai.UsbSpeed.HIGH and self._oak.device.getUsbSpeed() == dai.UsbSpeed.HIGH:
            warnings.warn("Device connected in USB2 mode! This might cause some issues. "
                          "In such case, please try using a (different) USB3 cable, "
                          "or force USB2 mode 'with OakCamera(usbSpeed='usb2') as oak:'", UsbWarning)

    def config_pipeline(self,
                        xlink_chunk: Optional[int] = None,
                        calib: Optional[dai.CalibrationHandler] = None,
                        tuning_blob: Optional[str] = None,
                        openvino_version: Union[None, str, dai.OpenVINO.Version] = None
                        ):
        """
        Configures DepthAI pipeline.
        @param xlink_chunk: Chunk size of XLink messages. 0 can result in lower latency
        @param calib: Calibration data to be uploaded to OAK
        @param tuning_blob: Camera tuning blob
        @param openvino_version: Force specific OpenVINO version
        """
        configPipeline(self.pipeline, xlink_chunk, calib, tuning_blob, openvino_version)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        logging.info("Closing OAK camera")
        if self.replay:
            logging.info("Closing replay")
            self.replay.close()
        if self._oak.device is not None:
            self._oak.device.close()

        for out in self._out_templates:
            if isinstance(out, RecordConfig):
                out.rec.close()
        self._oak.close()

    def start(self, blocking=False):
        """
        Start the application - upload the pipeline to the OAK device.
        Args:
            blocking (bool):  Continuously loop and call oak.poll() until program exits
        """
        self.build()

        # Remove unused nodes. There's a better way though.
        # self._pipeline.
        # schema = self._pipeline.serializeToJson()['pipeline']
        # used_nodes = []
        # for conn in schema['connections']:
        #     print()
        #     used_nodes.append(conn["node1Id"])
        #     used_nodes.append(conn["node2Id"])
        #
        # for node in self._pipeline.getAllNodes():
        #     if node.id not in used_nodes:
        #         print(f"Removed node {node} (id: {node.id}) from the pipeline as it hasn't been used!")
        #         self._pipeline.remove(node)

        self._oak.device.startPipeline(self.pipeline)

        self._oak.init_callbacks(self.pipeline)

        # Call on_pipeline_started() for each component
        for comp in self._components:
            comp.on_pipeline_started(self._oak.device)

        # Start FPS counters
        for xout in self._oak.oak_out_streams:  # Start FPS counters
            xout.start_fps()

        if self.replay:
            self.replay.createQueues(self._oak.device)
            # Called from Replay module on each new frame sent to the device.
            self.replay.start(self._oak.new_msg)

        # Check if callbacks (sync/non-sync are set)
        if blocking:
            # Constant loop: get messages, call callbacks
            while self.running():
                time.sleep(0.001)
                self.poll()

            cv2.destroyAllWindows()

    def running(self) -> bool:
        """
        Check if camera is running.
        Returns:
            True if camera is running, False otherwise.
        """
        return not self._stop

    def poll(self) -> Optional[int]:
        """
        Poll events; cv2.waitKey, send controls to OAK (if controls are enabled), update, check syncs.

        Returns: key pressed from cv2.waitKey, or None if
        """
        if CV2_HAS_GUI_SUPPORT:
            key = cv2.waitKey(1)
            if key == ord('q'):
                self._stop = True
                return key
        else:
            key = -1

        # TODO: check if components have controls enabled and check whether key == `control`

        self._oak.check_sync()

        if self.replay:
            if key == ord(' '):
                self.replay.toggle_pause()

            if self.replay._stop:
                self._stop = True
                return key

        for poll in self._polling:
            poll()  # Poll all callbacks

        if self.device.isClosed():
            self._stop = True
            return None

        return key

    def build(self) -> dai.Pipeline:
        """
        Connect to the device and build the pipeline based on previously provided configuration. Configure XLink queues,
        upload the pipeline to the device. This function must only be called once!  build() is also called by start().
        Return:
            Built dai.Pipeline
        """
        if self._built:
            return
        self._built = True

        # First go through each component to check whether any is forcing an OpenVINO version
        # TODO: check each component's SHAVE usage
        for c in self._components:
            ov = c.forced_openvino_version()
            if ov:
                if self.pipeline.getRequiredOpenVINOVersion() and self.pipeline.getRequiredOpenVINOVersion() != ov:
                    raise Exception(
                        'Two components forced two different OpenVINO version!'
                        'Please make sure that all your models are compiled using the same OpenVINO version.'
                    )
                self.pipeline.setOpenVINOVersion(ov)

        if self.pipeline.getRequiredOpenVINOVersion() is None:
            # Force 2021.4 as it's better supported (blobconverter, compile tool) for now.
            self.pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)


        # Create XLinkOuts based on visualizers/callbacks enabled

        # TODO: clean this up and potentially move elsewhere
        names = []
        for out in self._out_templates:
            xouts = out.setup(self.pipeline, self._oak.device, names)
            self._oak.oak_out_streams.extend(xouts)

        # User-defined arguments
        if self._args:
            self.config_pipeline(
                xlink_chunk=self._args.get('xlinkChunkSize', None),
                tuning_blob=self._args.get('cameraTuning', None),
                openvino_version=self._args.get('openvinoVersion', None),
            )

        return self.pipeline

    def _get_component_outputs(self, output: Union[List, Callable, Component]) -> List[Callable]:
        if not isinstance(output, List):
            output = [output]

        for i in range(len(output)):
            if isinstance(output[i], Component):
                # Select default (main) output of the component
                output[i] = output[i].out.main
        return output

    def sync(self, outputs: Union[Callable, List[Callable]], callback: Callable, visualize=False):
        """
        Synchronize multiple components outputs forward them to the callback.
        Args:
            outputs: Component output(s)
            callback: Where to send synced streams
            visualize: Whether to draw on the frames (like with visualize())
        """
        if isinstance(outputs, Callable):
            outputs = [outputs]  # to list

        self._out_templates.append(SyncConfig(outputs, callback))

    def record(self,
               outputs: Union[Callable, List[Callable]],
               path: str,
               record_type: RecordType = RecordType.VIDEO):
        """
        Record component outputs. This handles syncing multiple streams (eg. left, right, color, depth) and saving
        them to the computer in desired format (raw, mp4, mcap, bag..).
        Args:
            outputs (Component/Component output): Component output(s) to be recorded
            path: Folder path where to save these streams
            record_type: Record type
        """
        record = Record(Path(path).resolve(), record_type)
        self._out_templates.append(RecordConfig(self._get_component_outputs(outputs), record))
        return record

    def show_graph(self):
        """
        Shows DepthAI Pipeline graph, which can be useful when debugging. Builds the pipeline (oak.build()).
        """
        self.build()
        from depthai_pipeline_graph.pipeline_graph import \
            PipelineGraph

        p = PipelineGraph()
        p.create_graph(self.pipeline.serializeToJson()['pipeline'], self.device)
        self._polling.append(p.update)
        logging.info('Process started')

    def visualize(self,
                  output: Union[List, Callable, Component],
                  record_path: Optional[str] = None,
                  scale: float = None,
                  fps=False,
                  callback: Callable = None):
        """
        Visualize component output(s). This handles output streaming (OAK->host), message syncing, and visualizing.
        Args:
            output (Component/Component output): Component output(s) to be visualized. If component is passed, SDK will visualize its default output (out())
            record_path: Path where to store the recording (visualization window name gets appended to that path), supported formats: mp4, avi
            scale: Scale the output window by this factor
            fps: Whether to show FPS on the output window
            callback: Instead of showing the frame, pass the Packet to the callback function, where it can be displayed
        """
        if record_path and isinstance(output, List):
            if len(output) > 1:
                raise ValueError('Recording visualizer is only supported for a single output.')
            output = output[0]

        visualizer = Visualizer(scale, fps)
        return self._callback(output, callback, visualizer, record_path)

    def _callback(self,
                  output: Union[List, Callable, Component],
                  callback: Callable,
                  visualizer: Visualizer = None,
                  record_path: Optional[str] = None):
        if isinstance(output, List):
            for element in output:
                self._callback(element, callback, visualizer, record_path)
            return visualizer

        if isinstance(output, Component):
            output = output.out.main

        visualizer_enabled = visualizer is not None
        if visualizer_enabled:
            config = visualizer.config
            visualizer = copy.deepcopy(visualizer) or Visualizer()
            visualizer.config = config if config else visualizer.config

        self._out_templates.append(OutputConfig(output, callback, visualizer, visualizer_enabled, record_path))
        return visualizer

    def callback(self, output: Union[List, Callable, Component], callback: Callable, enable_visualizer: bool = False):
        """
        Create a callback for the component output(s). This handles output streaming (OAK->Host) and message syncing.
        Args:
            output: Component output(s) to be visualized. If component is passed, SDK will visualize its default output.
            callback: Handler function to which the Packet will be sent.
            enable_visualizer: Whether to enable visualizer for this output.
        """
        self._callback(output, callback, Visualizer() if enable_visualizer else None)

    def ros_stream(self, output: Union[List, Callable, Component]):
        self._out_templates.append(RosStreamConfig(self._get_component_outputs(output)))

    def trigger_action(self, trigger: Trigger, action: Union[Action, Callable]):
        self._out_templates.append(TriggerActionConfig(trigger, action))

    def set_max_queue_size(self, size: int):
        """
        Set maximum queue size for all outputs. This is the maximum number of frames that can be stored in the queue.
        Args:
            size: Maximum queue size for all outputs.
        """
        self._oak.set_max_queue_size(size)

    @property
    def device(self) -> dai.Device:
        """
        Returns dai.Device object. oak.built() has to be called before querying this property!
        """
        return self._oak.device

    @property
    def sensors(self) -> List[dai.CameraBoardSocket]:
        """
        Returns list of all sensors added to the pipeline.
        """
        return self._oak.image_sensors

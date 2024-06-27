import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

from depthai_sdk.logger import LOGGER
from depthai_sdk import CV2_HAS_GUI_SUPPORT
from depthai_sdk.types import Resolution
from depthai_sdk.visualize.visualizer import Visualizer

try:
    import cv2
except ImportError:
    cv2 = None

import depthai as dai

from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.args_parser import ArgsParser
from depthai_sdk.classes.packet_handlers import (
    BasePacketHandler,
    QueuePacketHandler,
    RosPacketHandler,
    TriggerActionPacketHandler,
    RecordPacketHandler,
    CallbackPacketHandler,
    VisualizePacketHandler
)
# RecordConfig, OutputConfig, SyncConfig, RosStreamConfig, TriggerActionConfig
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.components.imu_component import IMUComponent
from depthai_sdk.components.tof_component import ToFComponent
from depthai_sdk.components.nn_component import NNComponent
from depthai_sdk.components.parser import (
    parse_usb_speed,
    parse_camera_socket,
    get_first_color_cam,
    parse_open_vino_version
)
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.components.pointcloud_component import PointcloudComponent
from depthai_sdk.record import RecordType, Record
from depthai_sdk.replay import Replay
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger
from depthai_sdk.utils import report_crash_dump



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
                 replay: Union[None, str, Path] = None,
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
        self._pipeine_graph = None

        if args:
            if isinstance(args, bool):
                self._args = ArgsParser.parseArgs()
            else:  # Already parsed
                self._args = args
            # Set up the OakCamera
            if self._args.get('recording', None):
                replay = self._args.get('recording', None)
            if self._args.get('deviceId', None):
                device = self._args.get('deviceId', None)
            if self._args.get('usbSpeed', None):
                usb_speed = parse_usb_speed(self._args.get('usbSpeed', None))

            self.config_pipeline(
                xlink_chunk=self._args.get('xlinkChunkSize', None),
                tuning_blob=self._args.get('cameraTuning', None),
                openvino_version=self._args.get('openvinoVersion', None),
            )

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
        self._packet_handlers: List[BasePacketHandler] = []

        self._rotation = rotation
        if replay is not None:
            self.replay = Replay(replay)
            self.replay.initPipeline(self.pipeline)
            LOGGER.info(f'Available streams from recording: {self.replay.getStreams()}')
        self._calibration = self._init_calibration()

    def camera(self,
               source: Union[str, dai.CameraBoardSocket],
               resolution: Optional[Union[
                   str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
               ]] = None,
               fps: Optional[float] = None,
               encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
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
        """
        sensor_type = None
        if isinstance(source, str):
            if "," in source:  # For sensors that support multiple
                parts = source.lower().split(',')
                source = parts[0]
                if parts[1] in ["c", "color"]:
                    sensor_type = dai.CameraSensorType.COLOR
                elif parts[1] in ["m", "mono"]:
                    sensor_type = dai.CameraSensorType.MONO
                else:
                    raise Exception(
                        "Please specify sensor type with c/color or m/mono after the ','"
                        " - eg. `cam = oak.create_camera('cama,c')`"
                    )

            if source == 'left':
                source = self._calibration.getStereoLeftCameraId()
            elif source == 'right':
                source = self._calibration.getStereoRightCameraId()
            elif source in ['color', 'rgb']:
                source = get_first_color_cam(self.device)
            else:
                source = parse_camera_socket(source)

            if source in [None, dai.CameraBoardSocket.AUTO]:
                return None  # There's no camera on this socket

        for comp in self._components:
            if isinstance(comp, CameraComponent) and comp._socket == source:
                return comp

        comp = CameraComponent(self.device,
                               self.pipeline,
                               source=source,
                               resolution=resolution,
                               fps=fps,
                               encode=encode,
                               sensor_type=sensor_type,
                               rotation=self._rotation,
                               replay=self.replay,
                               args=self._args)
        self._components.append(comp)
        return comp

    def _init_device(self,
                     config: dai.Device.Config,
                     device_str: Optional[str] = None,
                     ) -> None:

        """
        Connect to the OAK camera
        """
        self.device = None
        if device_str is not None:
            device_info = dai.DeviceInfo(device_str)
        else:
            (found, device_info) = dai.Device.getFirstAvailableDevice()
            if not found:
                raise Exception("No OAK device found to connect to!")

        self.device = dai.Device(
            config=config,
            deviceInfo=device_info,
        )

        # TODO test with usb3 (SUPER speed)
        if config.board.usb.maxSpeed != dai.UsbSpeed.HIGH and self.device.getUsbSpeed() == dai.UsbSpeed.HIGH:
            warnings.warn("Device connected in USB2 mode! This might cause some issues. "
                          "In such case, please try using a (different) USB3 cable, "
                          "or force USB2 mode 'with OakCamera(usb_speed='usb2') as oak:'", UsbWarning)

    def create_camera(self,
                      source: Union[str, dai.CameraBoardSocket],
                      resolution: Optional[Resolution] = None,
                      fps: Optional[float] = None,
                      encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
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
        """
        return self.camera(source, resolution, fps, encode)

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
        if self.replay:
            sources = self.replay.getStreams()  # TODO handle in case the stream is not from a camera
        else:
            sources = [cam_sensor.socket for cam_sensor in self.device.getConnectedCameraFeatures()]
        for source in sources:
            comp = CameraComponent(self.device,
                                   self.pipeline,
                                   source=source,
                                   resolution=resolution,
                                   fps=fps,
                                   encode=encode,
                                   rotation=self._rotation,
                                   replay=self.replay,
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
        """
        if spatial and type(spatial) == bool:
            spatial = self.stereo()

        comp = NNComponent(self.device,
                           self.pipeline,
                           model=model,
                           input=input,
                           nn_type=nn_type,
                           tracker=tracker,
                           spatial=spatial,
                           decode_fn=decode_fn,
                           replay=self.replay,
                           args=self._args)
        self._components.append(comp)
        return comp

    def stereo(self,
               resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
               fps: Optional[float] = None,
               left: Union[None, dai.Node.Output, CameraComponent] = None,  # Left mono camera
               right: Union[None, dai.Node.Output, CameraComponent] = None,  # Right mono camera
               encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None
               ) -> StereoComponent:
        """
        Create Stereo camera component. If left/right cameras/component aren't specified they will get created internally.

        Args:
            resolution (str/SensorResolution): If monochrome cameras aren't already passed, create them and set specified resolution
            fps (float): If monochrome cameras aren't already passed, create them and set specified FPS
            left (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
            right (CameraComponent/dai.node.MonoCamera): Pass the camera object (component/node) that will be used for stereo camera.
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via StereoComponent.out.encoded). If True, it will use MJPEG.
        """
        if left is None:
            left = self.camera(source="left", resolution=resolution, fps=fps)
        if right is None:
            right = self.camera(source="right", resolution=resolution, fps=fps)

        if right is None or left is None:
            return None

        comp = StereoComponent(self.device,
                               self.pipeline,
                               left=left,
                               right=right,
                               replay=self.replay,
                               args=self._args,
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
            encode (bool/str/Profile): Whether we want to enable video encoding (accessible via StereoComponent.out.encoded). If True, it will use h264 codec.
        """
        return self.stereo(resolution, fps, left, right, encode)

    def create_tof(self,
                   source: Union[str, dai.CameraBoardSocket, None] = None,
                   fps: Optional[float] = None,
                   align_to: Optional[CameraComponent] = None) -> ToFComponent:
        """
        Create ToF component.

        Args:
            source (str / dai.CameraBoardSocket): ToF camera source
            fps (float): Sensor FPS
            align_to (CameraComponent): Align ToF to this camera component
        """
        comp = ToFComponent(self.device, self.pipeline, source, align_to, fps)
        self._components.append(comp)
        return comp

    def create_imu(self) -> IMUComponent:
        """
        Create IMU component
        """
        comp = IMUComponent(self.device, self.pipeline)
        self._components.append(comp)
        return comp

    def create_pointcloud(self,
                          depth_input: Union[None, StereoComponent, ToFComponent, dai.node.StereoDepth, dai.Node.Output] = None,
                          colorize: Union[None, CameraComponent, dai.node.MonoCamera, dai.node.ColorCamera, dai.Node.Output, bool] = None,
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
            self.device,
            self.pipeline,
            depth_input=depth_input,
            colorize=colorize,
            replay=self.replay,
            args=self._args,
        )
        self._components.append(comp)
        return comp

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
        if xlink_chunk is not None:
            self.pipeline.setXLinkChunkSize(xlink_chunk)
        if calib is not None:
            self.pipeline.setCalibrationData(calib)
        if tuning_blob is not None:
            self.pipeline.setCameraTuningBlobPath(tuning_blob)
        ov_version = parse_open_vino_version(openvino_version)
        if ov_version is not None:
            self.pipeline.setOpenVINOVersion(ov_version)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def close(self):
        LOGGER.info("Closing OAK camera")
        if self.replay:
            self.replay.close()

        for handler in self._packet_handlers:
            handler.close()

        self.device.close()

    def _new_oak_msg(self, q_name: str, msg):
        if self._stop:
            return
        if q_name in self._new_msg_callbacks:
            for callback in self._new_msg_callbacks[q_name]:
                callback(q_name, msg)

    def start(self, blocking=False):
        """
        Start the application - upload the pipeline to the OAK device.
        Args:
            blocking (bool):  Continuously loop and call oak.poll() until program exits
        """
        self._new_msg_callbacks = {}
        for node in self.pipeline.getAllNodes():
            if isinstance(node, dai.node.XLinkOut):
                self._new_msg_callbacks[node.getStreamName()] = []

        for handler in self._packet_handlers:
            # Setup PacketHandlers. This will:
            # - Initialize all submodules (eg. Recording, Trigger/Actions, Visualizer)
            # - Create XLinkIn nodes for all components/streams
            handler.setup(self.pipeline, self.device, self._new_msg_callbacks)

        # Upload the pipeline to the device and start it
        self.device.startPipeline(self.pipeline)

        for xlink_name in self._new_msg_callbacks:
            try:
                self.device.getOutputQueue(xlink_name, maxSize=1, blocking=False).addCallback(self._new_oak_msg)
            # TODO: make this nicer, have self._new_msg_callbacks know whether it's replay or not
            except Exception as e:
                if self.replay:
                    self.replay._add_callback(xlink_name, self._new_oak_msg)
                else:
                    raise e

        # Append callbacks to be called from main thread
        # self._polling.append()
        if self._pipeine_graph is not None:
            self._pipeine_graph.create_graph(self.pipeline.serializeToJson()['pipeline'], self.device)
            LOGGER.info('Pipeline graph process started')

        # Call on_pipeline_started() for each component
        for comp in self._components:
            comp.on_pipeline_started(self.device)

        if self.replay:
            self.replay.createQueues(self.device)
            self.replay.start()
            # Called from Replay module on each new frame sent to the device.

        # Check if callbacks (sync/non-sync are set)
        if blocking:
            # Constant loop: get messages, call callbacks
            while self.running():
                self.poll()

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
        # if self._stop:
        #     return
        if CV2_HAS_GUI_SUPPORT:
            key = cv2.waitKey(1)
            if key == ord('q'):
                self._stop = True
                return key
        else:
            time.sleep(0.001)
            key = -1

        # TODO: check if components have controls enabled and check whether key == `control`
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

    def sync(self, outputs: Union[ComponentOutput, List[ComponentOutput]], callback: Callable, visualize=False):
        raise DeprecationWarning('sync() is deprecated. Use callback() instead.')

    def record(self,
               outputs: Union[ComponentOutput, List[ComponentOutput]],
               path: str,
               record_type: RecordType = RecordType.VIDEO
               ) -> RecordPacketHandler:
        """
        Record component outputs. This handles syncing multiple streams (eg. left, right, color, depth) and saving
        them to the computer in desired format (raw, mp4, mcap, bag..).

        Args:
            outputs (Component/Component output): Component output(s) to be recorded.
            path: Folder path where to save these streams.
            record_type: Record type.
        """
        handler = RecordPacketHandler(outputs, Record(Path(path).resolve(), record_type))
        self._packet_handlers.append(handler)
        return handler

    def show_graph(self):
        """
        Shows DepthAI Pipeline graph, which can be useful when debugging. You must call this BEFORE calling the oak.start()!
        """
        from depthai_pipeline_graph.pipeline_graph import PipelineGraph
        self._pipeine_graph = PipelineGraph()
        self._polling.append(self._pipeine_graph.update)

    def visualize(self,
                  output: Union[List, ComponentOutput, Component],
                  record_path: Optional[str] = None,
                  scale: float = None,
                  fps=False,
                  callback: Callable = None,
                  visualizer: str = 'opencv'
                  ) -> Visualizer:
        """
        Visualize component output(s). This handles output streaming (OAK->host), message syncing, and visualizing.

        Args:
            output (Component/Component output): Component output(s) to be visualized. If component is passed, SDK will visualize its default output (out()).
            record_path: Path where to store the recording (visualization window name gets appended to that path), supported formats: mp4, avi.
            scale: Scale the output window by this factor.
            fps: Whether to show FPS on the output window.
            callback: Instead of showing the frame, pass the Packet to the callback function, where it can be displayed.
            visualizer: Which visualizer to use. Options: 'opencv', 'depthai-viewer', 'robothub'.
        """
        main_thread = False
        visualizer = visualizer.lower()
        if visualizer in ['opencv', 'cv2']:
            from depthai_sdk.visualize.visualizers.opencv_visualizer import OpenCvVisualizer
            vis = OpenCvVisualizer(scale, fps)
            main_thread = True  # OpenCV's imshow() requires to be called from the main thread
        elif visualizer in ['depthai-viewer', 'depthai_viewer', 'viewer', 'depthai']:
            from depthai_sdk.visualize.visualizers.viewer_visualizer import DepthaiViewerVisualizer
            vis = DepthaiViewerVisualizer(scale, fps)
        elif visualizer in ['robothub', 'rh']:
            raise NotImplementedError('Robothub visualizer is not implemented yet')
        else:
            raise ValueError(f"Unknown visualizer: {visualizer}. Options: 'opencv'")

        handler = VisualizePacketHandler(output,
                                         vis,
                                         callback=callback, record_path=record_path,
                                         main_thread=main_thread)
        self._packet_handlers.append(handler)

        if main_thread:
            self._polling.append(handler._poll)

        return vis

    def queue(self, output: Union[ComponentOutput, Component, List], max_size: int = 30) -> QueuePacketHandler:
        """
        Create a queue for the component output(s). This handles output streaming (OAK->Host) and message syncing.

        Args:
            output: Component output(s) to be visualized. If component is passed, SDK will visualize its default output.
            max_size: Maximum queue size for this output.
        """
        handler = QueuePacketHandler(output, max_size)
        self._packet_handlers.append(handler)
        return handler

    def callback(self,
                 output: Union[List, Callable, Component],
                 callback: Callable,
                 main_thread=False
                 ) -> CallbackPacketHandler:
        """
        Create a callback for the component output(s). This handles output streaming (OAK->Host) and message syncing.

        Args:
            output: Component output(s) to be visualized. If component is passed, SDK will visualize its default output.
            callback: Handler function to which the Packet will be sent.
            main_thread: Whether to run the callback in the main thread. If False, it will call the callback in a separate thread, so some functions (eg. cv2.imshow) won't work.
        """
        handler = CallbackPacketHandler(output, callback=callback, main_thread=main_thread)
        if main_thread:
            self._polling.append(handler._poll)
        self._packet_handlers.append(handler)
        return handler

    def ros_stream(self, output: Union[List, ComponentOutput, Component]) -> RosPacketHandler:
        """
        Publish component output(s) to ROS streams.
        """
        handler = RosPacketHandler(output)
        self._packet_handlers.append(handler)
        return handler

    def trigger_action(self, trigger: Trigger, action: Union[Action, Callable]) -> None:
        self._packet_handlers.append(TriggerActionPacketHandler(trigger, action))

    @property
    def sensors(self) -> List[dai.CameraBoardSocket]:
        """
        Returns list of all sensors added to the pipeline.
        """
        return self.device.getConnectedCameraFeatures()

    def _init_calibration(self) -> dai.CalibrationHandler:
        if self.replay:
            calibration = self.pipeline.getCalibrationData()
        else:
            calibration = self.device.readCalibration()
        if calibration is None:
            LOGGER.warning("No calibration data found on the device or in replay")
        return calibration

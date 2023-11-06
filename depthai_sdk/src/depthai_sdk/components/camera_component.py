from typing import Dict

from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.components.camera_control import CameraControl
from depthai_sdk.components.camera_helper import *
from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.components.parser import parse_resolution, parse_encode, encoder_profile_to_fourcc
from depthai_sdk.logger import LOGGER
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout, ReplayStream
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.replay import Replay
from depthai_sdk.types import Resolution


class CameraComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 source: dai.CameraBoardSocket,
                 resolution: Optional[Union[
                     str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                 ]] = None,
                 fps: Optional[float] = None,
                 encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                 sensor_type: Optional[dai.CameraSensorType] = None,
                 rotation: Optional[int] = None,
                 replay: Optional[Replay] = None,
                 name: Optional[str] = None,
                 args: Dict = None):
        """
        Creates Camera component. This abstracts ColorCamera/MonoCamera nodes and supports mocking the camera when
        recording is passed during OakCamera initialization. Mocking the camera will send frames from the host to the
        OAK device (via XLinkIn node).

        Args:
            device (dai.Device): OAK device
            pipeline (dai.Pipeline): OAK pipeline
            source (str or dai.CameraBoardSocket): Source of the camera. Either color/rgb/right/left
            resolution (optional): Camera resolution, eg. '800p' or '4k'
            fps (float, optional): Camera FPS
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            sensor_type: To force color/mono/tof camera
            rotation (int, optional): Rotate the camera by 90, 180, 270 degrees
            replay (Replay object): Replay object to use for mocking the camera
            name (str, optional): Name of the output stream
            args (Dict): Use user defined arguments when constructing the pipeline
        """
        super().__init__()
        # _replay should be set before .out, as it's used in .out
        self._replay: Optional[Replay] = replay
        self.out = self.Out(self)

        self._pipeline = pipeline
        self._device = device

        self.node: Optional[Union[dai.node.ColorCamera, dai.node.MonoCamera, dai.node.XLinkIn]] = None
        self.encoder: Optional[dai.node.VideoEncoder] = None

        self.stream: Optional[dai.Node.Output] = None  # Node output to be used as eg. an input into NN
        self.stream_size: Optional[Tuple[int, int]] = None  # Output size

        self._source = str(source)
        if self._source.startswith('CameraBoardSocket.'):
            self._source = self._source[len('CameraBoardSocket.'):]

        self._socket = source
        self._replay: Optional[Replay] = replay
        self._args: Dict = args

        self.name = name

        if rotation not in [None, 0, 90, 180, 270]:
            raise ValueError(f'Angle {rotation} not supported! Use 0, 90, 180, 270.')

        self._num_frames_pool = 10
        self._preview_num_frames_pool = 4

        if self.is_replay():
            stream_name = None
            for name, stream in self._replay.streams.items():
                if stream.get_socket() == self._socket:
                    stream_name = name
                    break
            if stream_name is None:
                raise Exception(f"{source} stream was not found in specified depthai-recording!")
            self._source = stream_name
            res = self._replay.getShape(self._source)
            # print('resolution', res)
            # resize = getResize(res, width=1200)
            # self._replay.setResizeColor(resize)
            stream = self._replay.streams[self._source]
            if stream.node is None:
                return  # Stream disabled

            self.node = stream.node
            # print('resize', resize)
            self.node.setMaxDataSize(res[0] * res[1] * 3)
            self.stream_size = res
            self.stream = self.node.out

            if rotation in [90, 180, 270]:
                rot_manip = self._create_rotation_manip(pipeline, rotation)
                self.node.out.link(rot_manip.inputImage)
                self.stream = rot_manip.out
                if rotation in [90, 270]:
                    self.stream_size = self.stream_size[::-1]
        # Livestreaming, not replay
        else:
            node_type: dai.node = None
            sensors = [f for f in device.getConnectedCameraFeatures() if f.socket == source]
            if len(sensors) == 0:
                raise Exception(f"No camera found on user-specified socket {source}")
            sensor = sensors[0]

            sensor_type = sensor_type or sensor.supportedTypes[0]
            if sensor_type == dai.CameraSensorType.COLOR:
                node_type = dai.node.ColorCamera
            elif sensor_type == dai.CameraSensorType.MONO:
                node_type = dai.node.MonoCamera
            else:
                raise Exception(f"{sensor} doesn't support either COLOR or MONO ")

            # Create the node, and set the socket
            self.node = pipeline.create(node_type)
            self.node.setBoardSocket(source)

        self._resolution_forced: bool = resolution is not None
        if resolution:
            self._set_resolution(resolution)
        if fps:
            self.set_fps(fps)

        # Default configuration for the nodes
        if isinstance(self.node, dai.node.ColorCamera):
            # DepthAI uses CHW (Planar) channel layout convention for NN inferencing
            self.node.setInterleaved(False)
            # DepthAI uses BGR color order convention for NN inferencing
            self.node.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            self.node.setPreviewNumFramesPool(self._preview_num_frames_pool)

            if not self._resolution_forced:  # Find the closest resolution
                sensor = [f for f in device.getConnectedCameraFeatures() if f.socket == self.node.getBoardSocket()][0]
                sensor_type = dai.CameraSensorType.COLOR if dai.node.ColorCamera else dai.CameraSensorType.MONO
                targetWidthRes = 1300
                targetWidthIsp = targetWidthRes
                if self._args["defaultResolution"] == "min":
                    targetWidthRes = 0
                    targetWidthIsp = 1300  # Still keep the same target for the ISP
                elif self._args["defaultResolution"] == "max":
                    targetWidthRes = 1000000  # Some big number
                    targetWidthIsp = targetWidthRes
                res = getClosesResolution(sensor, sensor_type, width=targetWidthRes)
                self.node.setResolution(res)
                scale = getClosestIspScale(self.node.getIspSize(), width=targetWidthIsp,
                                           videoEncoder=(encode is not None))
                self.node.setIspScale(*scale)

            curr_size = self.node.getVideoSize()
            closest = getClosestVideoSize(*curr_size, videoEncoder=encode)
            self.node.setVideoSize(*closest)
            self.node.setVideoNumFramesPool(2)  # We will increase it later if we are streaming to host

            self.node.setPreviewSize(*self.node.getVideoSize())
            self.stream_size = self.node.getPreviewSize()
            self.stream = self.node.preview

        elif isinstance(self.node, dai.node.MonoCamera):
            self.stream_size = self.node.getResolutionSize()
            self.stream = self.node.out

        if rotation and not self.is_replay():
            if rotation == 180:
                self.node.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
            else:
                rot_manip = self._create_rotation_manip(pipeline, rotation)
                self.stream.link(rot_manip.inputImage)
                self.stream = rot_manip.out
                self.stream_size = self.stream_size[::-1]

        if encode:
            self.encoder = pipeline.createVideoEncoder()
            self._encoder_profile = parse_encode(encode)  # MJPEG by default
            self.encoder.setDefaultProfilePreset(self.get_fps(), self._encoder_profile)

            if self.is_replay():  # TODO - this might be not needed, we check for replay above and return
                # Create ImageManip to convert to NV12
                type_manip = pipeline.createImageManip()
                type_manip.setFrameType(dai.ImgFrame.Type.NV12)
                type_manip.setMaxOutputFrameSize(self.stream_size[0] * self.stream_size[1] * 3)

                self.stream.link(type_manip.inputImage)
                type_manip.out.link(self.encoder.input)
            elif self.is_mono():
                self.stream.link(self.encoder.input)
            elif self.is_color():
                self.node.video.link(self.encoder.input)
            else:
                raise ValueError('CameraComponent is neither Color, Mono, nor Replay!')

        if self._args:
            self._config_camera_args(self._args)

        # Runtime camera control
        self.control = CameraControl()
        self._control_xlink_in = None
        if not self.is_replay():
            self._control_xlink_in = pipeline.create(dai.node.XLinkIn)
            self._control_xlink_in.setStreamName(f"{self.node.id}_inputControl")
            self._control_xlink_in.out.link(self.node.inputControl)
            # CameraControl message doesn't use any additional data (only metadata)
            self._control_xlink_in.setMaxDataSize(1)

    def on_pipeline_started(self, device: dai.Device):
        if self._control_xlink_in is not None:
            queue = device.getInputQueue(self._control_xlink_in.getStreamName())
            self.control.set_input_queue(queue)

    def _create_rotation_manip(self, pipeline: dai.Pipeline, rotation: int):
        rot_manip = pipeline.createImageManip()
        rgb_rr = dai.RotatedRect()
        w, h = self.stream_size
        rgb_rr.center.x, rgb_rr.center.y = w // 2, h // 2
        rgb_rr.size.width, rgb_rr.size.height = (w, h) if rotation % 180 == 0 else (h, w)
        rgb_rr.angle = rotation
        rot_manip.initialConfig.setCropRotatedRect(rgb_rr, False)
        rot_manip.setMaxOutputFrameSize(w * h * 3)
        return rot_manip

    def config_camera(self,
                      # preview: Union[None, str, Tuple[int, int]] = None,
                      size: Union[None, Tuple[int, int], str] = None,
                      resize_mode: ResizeMode = ResizeMode.CROP,
                      fps: Optional[float] = None,
                      resolution: Optional[Resolution] = None
                      ) -> None:
        """
        Configure resolution, scale, FPS, etc.
        """
        # TODO: Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
        if fps:
            self.set_fps(fps)
        if resolution:
            self._set_resolution(resolution)

        if size:
            from .parser import parse_size
            size_tuple = parse_size(size)

            if self._replay:
                self._replay.resize(self._source, size_tuple, resize_mode)
            elif self.is_color():
                self.node.setStillSize(*size_tuple)
                self.node.setVideoSize(*size_tuple)
                self.node.setPreviewSize(*size_tuple)
                if resize_mode != ResizeMode.CROP:
                    raise ValueError("Currently only ResizeMode.CROP is supported mode for specifying size!")
            else:
                # TODO: Use ImageManip to set mono frame size
                raise NotImplementedError("Not yet implemented")

    def _config_camera_args(self, args: Dict):
        if not isinstance(args, Dict):
            args = vars(args)  # Namespace -> Dict

        if self.is_color():
            self.config_camera(
                fps=args.get('rgbFps', None),
                resolution=args.get('rgbResolution', None),
            )
            self.config_color_camera(
                manual_focus=args.get('manualFocus', None),
                af_mode=args.get('afMode', None),
                awb_mode=args.get('awbMode', None),
                scene_mode=args.get('sceneMode', None),
                anti_banding_mode=args.get('antiBandingMode', None),
                effect_mode=args.get('effectMode', None),
                isp_scale=args.get('ispScale', None),
                sharpness=args.get('sharpness', None),
                luma_denoise=args.get('lumaDenoise', None),
                chroma_denoise=args.get('chromaDenoise', None),
            )
        elif self.is_mono():
            self.config_camera(
                fps=args.get('monoFps', None),
                resolution=args.get('monoResolution', None),
            )
        else:  # Replay
            self.config_camera(fps=args.get('fps', None))

    def control_with_nn(self, detection_component: 'NNComponent', auto_focus=True, auto_exposure=True, debug=False):
        """
        Control the camera AF/AE/AWB based on the object detection results.

        :param detection_component: NNComponent that will be used to control the camera
        :param auto_focus: Enable auto focus to the object
        :param auto_exposure: Enable auto exposure to the object
        """

        if not auto_focus and not auto_exposure:
            LOGGER.error('Attempted to control camera with NN, '
                          'but both Auto-Focus and Auto-Exposure were disabled! Attempt ignored.')
            return

        if 'NNComponent' not in str(type(detection_component)):
            raise ValueError('nn_component must be an instance of NNComponent!')
        if not detection_component.is_detector():
            raise ValueError('nn_component must be a object detection model (YOLO/MobileNetSSD based)!')

        from depthai_sdk.components.control_camera_with_nn import control_camera_with_nn

        control_camera_with_nn(
            pipeline=self._pipeline,
            camera_control=self.node.inputControl,
            nn_output=detection_component.node.out,
            resize_mode=detection_component._ar_resize_mode,
            resolution=self.node.getResolution(),
            nn_size=detection_component._size,
            af=auto_focus,
            ae=auto_exposure,
            debug=debug
        )

    def config_color_camera(self,
                            interleaved: Optional[bool] = None,
                            color_order: Union[None, dai.ColorCameraProperties.ColorOrder, str] = None,
                            # Cam control
                            manual_focus: Optional[int] = None,
                            af_mode: Optional[dai.CameraControl.AutoFocusMode] = None,
                            awb_mode: Optional[dai.CameraControl.AutoWhiteBalanceMode] = None,
                            scene_mode: Optional[dai.CameraControl.SceneMode] = None,
                            anti_banding_mode: Optional[dai.CameraControl.AntiBandingMode] = None,
                            effect_mode: Optional[dai.CameraControl.EffectMode] = None,
                            # IQ settings
                            isp_scale: Optional[Tuple[int, int]] = None,
                            sharpness: Optional[int] = None,
                            luma_denoise: Optional[int] = None,
                            chroma_denoise: Optional[int] = None,
                            ) -> None:
        if not self.is_color():
            LOGGER.info('Attempted to configure ColorCamera, '
                         'but this component doesn\'t have it. Config attempt ignored.')
            return

        if self.is_replay():
            LOGGER.info('Tried configuring ColorCamera, but replaying is enabled. Config attempt ignored.')
            return

        if interleaved is not None: self.node.setInterleaved(interleaved)
        if color_order:
            if isinstance(color_order, str):
                color_order = getattr(dai.ColorCameraProperties.ColorOrder, color_order.upper())
            self.node.setColorOrder(color_order)

        if manual_focus is not None: self.node.initialControl.setManualFocus(manual_focus)
        if af_mode: self.node.initialControl.setAutoFocusMode(af_mode)
        if awb_mode: self.node.initialControl.setAutoWhiteBalanceMode(awb_mode)
        if scene_mode: self.node.initialControl.setSceneMode(scene_mode)
        if anti_banding_mode: self.node.initialControl.setAntiBandingMode(anti_banding_mode)
        if effect_mode: self.node.initialControl.setEffectMode(effect_mode)
        # EQ settings
        if isp_scale:
            self._resolution_forced = True
            self.node.setIspScale(*isp_scale)

            self.node.setPreviewSize(*self.node.getIspSize())
            self.node.setVideoSize(*self.node.getIspSize())
            self.stream_size = self.node.getIspSize()
            self.stream = self.node.preview

        if sharpness is not None: self.node.initialControl.setSharpness(sharpness)
        if luma_denoise is not None: self.node.initialControl.setLumaDenoise(luma_denoise)
        if chroma_denoise is not None: self.node.initialControl.setChromaDenoise(chroma_denoise)

    def _set_resolution(self, resolution):
        if not self.is_replay():
            if isinstance(resolution, str) and resolution.lower() in ['max', 'maximum']:
                sensor = [f for f in self._device.getConnectedCameraFeatures() if f.socket == self._socket][0]
                resolution = get_max_resolution(type(self.node), sensor)
            else:
                resolution = parse_resolution(type(self.node), resolution)
            self.node.setResolution(resolution)
        # TODO: support potentially downscaling depthai-recording

    def is_replay(self) -> bool:
        return self._replay is not None

    def is_color(self) -> bool:
        return isinstance(self.node, dai.node.ColorCamera)

    def is_mono(self) -> bool:
        return isinstance(self.node, dai.node.MonoCamera)

    def get_fps(self) -> float:
        if self.is_replay():
            return self._replay.get_fps()
        else:
            return self.node.getFps()

    def set_fps(self, fps: float):
        if self.is_replay():
            self._replay.set_fps(fps)
        else:
            self.node.setFps(fps)

    def config_encoder_h26x(self,
                            rate_control_mode: Optional[dai.VideoEncoderProperties.RateControlMode] = None,
                            keyframe_freq: Optional[int] = None,
                            bitrate_kbps: Optional[int] = None,
                            num_b_frames: Optional[int] = None,
                            ):
        if self.encoder is None:
            raise Exception('Video encoder was not enabled!')
        if self._encoder_profile == dai.VideoEncoderProperties.Profile.MJPEG:
            raise Exception('Video encoder was set to MJPEG while trying to configure H26X attributes!')

        if rate_control_mode is not None:
            self.encoder.setRateControlMode(rate_control_mode)
        if keyframe_freq is not None:
            self.encoder.setKeyframeFrequency(keyframe_freq)
        if bitrate_kbps is not None:
            self.encoder.setBitrateKbps(bitrate_kbps)
        if num_b_frames is not None:
            self.encoder.setNumBFrames(num_b_frames)

    def config_encoder_mjpeg(self,
                             quality: Optional[int] = None,
                             lossless: bool = False
                             ):
        if self.encoder is None:
            raise Exception('Video encoder was not enabled!')
        if self._encoder_profile != dai.VideoEncoderProperties.Profile.MJPEG:
            raise Exception(
                f'Video encoder was set to {self._encoder_profile} while trying to configure MJPEG attributes!'
            )

        if quality is not None:
            self.encoder.setQuality(quality)
        if lossless is not None:
            self.encoder.setLossless(lossless)

    def get_stream_xout(self, fourcc: Optional[str] = None) -> StreamXout:
        if self.encoder is not None and fourcc is not None:
            return StreamXout(self.encoder.bitstream, name=self.name or self._source + '_bitstream')
        elif self.is_replay():
            return ReplayStream(self.name or self._source)
        elif self.is_mono():
            return StreamXout(self.stream, name=self.name or self._source + '_mono')
        else:  # ColorCamera
            self.node.setVideoNumFramesPool(self._num_frames_pool)
            self.node.setPreviewNumFramesPool(self._preview_num_frames_pool)
            # node.video instead of preview (self.stream) was used to reduce bandwidth
            # consumption by 2 (3bytes/pixel vs 1.5bytes/pixel)
            return StreamXout(self.node.video, name=self.name or self._source + '_video')

    def set_num_frames_pool(self, num_frames: int, preview_num_frames: Optional[int] = None):
        """
        Set the number of frames to be stored in the pool.

        :param num_frames: Number of frames to be stored in the pool.
        :param preview_num_frames: Number of frames to be stored in the pool for the preview stream.
        """
        if self.is_color():
            self._num_frames_pool = num_frames
            if preview_num_frames is not None:
                self._preview_num_frames_pool = preview_num_frames

    def get_fourcc(self) -> Optional[str]:
        if self.encoder is None:
            return None
        return encoder_profile_to_fourcc(self._encoder_profile)

    """
    Available outputs (to the host) of this component
    """

    class Out:
        class CameraOut(ComponentOutput):
            def __call__(self, device: dai.Device, fourcc: Optional[str] = None) -> XoutBase:
                return XoutFrames(self._comp.get_stream_xout(fourcc), fourcc).set_comp_out(self)

        class ReplayOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                return XoutFrames(ReplayStream(self._comp._source)).set_comp_out(self)

        class EncodedOut(CameraOut):
            def __call__(self, device: dai.Device) -> XoutBase:
                return super().__call__(device, fourcc=self._comp.get_fourcc())


        def __init__(self, camera_component: 'CameraComponent'):
            self.replay = self.ReplayOut(camera_component)
            self.camera = self.CameraOut(camera_component)
            self.encoded = self.EncodedOut(camera_component)

            self.main = self.replay if camera_component.is_replay() else self.camera

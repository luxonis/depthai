import logging
from typing import Dict

from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.components.camera_helper import *
from depthai_sdk.components.component import Component
from depthai_sdk.components.parser import parse_resolution, parse_encode, parse_camera_socket
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout, ReplayStream
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_h26x import XoutH26x
from depthai_sdk.oak_outputs.xout.xout_mjpeg import XoutMjpeg
from depthai_sdk.replay import Replay
from depthai_sdk.components.camera_control import CameraControl


class CameraComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 source: Union[str, dai.CameraBoardSocket],
                 resolution: Optional[Union[
                     str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                 ]] = None,
                 fps: Optional[float] = None,
                 encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
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
            rotation (int, optional): Rotate the camera by 90, 180, 270 degrees
            replay (Replay object): Replay object to use for mocking the camera
            name (str, optional): Name of the output stream
            args (Dict): Use user defined arguments when constructing the pipeline
        """
        super().__init__()
        self.out = self.Out(self)
        self._pipeline = pipeline

        self.node: Optional[Union[dai.node.ColorCamera, dai.node.MonoCamera, dai.node.XLinkIn]] = None
        self.encoder: Optional[dai.node.VideoEncoder] = None

        self.stream: Optional[dai.Node.Output] = None  # Node output to be used as eg. an input into NN
        self.stream_size: Optional[Tuple[int, int]] = None  # Output size

        self._source = str(source)
        self._replay: Optional[Replay] = replay
        self._args: Dict = args
        self.name = name

        if rotation not in [None, 0, 90, 180, 270]:
            raise ValueError(f'Angle {rotation} not supported! Use 0, 90, 180, 270.')

        self._num_frames_pool = 10
        self._preview_num_frames_pool = 4

        if self.is_replay():
            if source.casefold() not in list(map(lambda x: x.casefold(), self._replay.getStreams())):
                raise Exception(f"{source} stream was not found in specified depthai-recording!")
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
            if isinstance(source, str):
                source = source.upper()
                # When sensors can be either color or mono (eg. AR0234), we allow specifying it
                if "," in source:  # For sensors that support multiple
                    parts = source.split(',')
                    source = parts[0]
                    if parts[1] in ["C", "COLOR"]:
                        node_type = dai.node.ColorCamera
                    elif parts[1] in ["M", "MONO"]:
                        node_type = dai.node.MonoCamera
                    else:
                        raise Exception(
                            "Please specify sensor type with c/color or m/mono after the ','"
                            " - eg. `cam = oak.create_camera('cama,c')`"
                        )
                elif source in ["COLOR", "RGB"]:
                    for features in device.getConnectedCameraFeatures():
                        if dai.CameraSensorType.COLOR in features.supportedTypes:
                            source = features.socket
                            break
                    if not isinstance(source, dai.CameraBoardSocket):
                        raise ValueError("Couldn't find a color camera!")

            socket = parse_camera_socket(source)
            sensor = [f for f in device.getConnectedCameraFeatures() if f.socket == socket][0]

            if node_type is None:  # User specified camera type
                type = sensor.supportedTypes[0]
                if type == dai.CameraSensorType.COLOR:
                    node_type = dai.node.ColorCamera
                elif type == dai.CameraSensorType.MONO:
                    node_type = dai.node.MonoCamera
                else:
                    raise Exception(f"{sensor} doesn't support either COLOR or MONO ")

            # Create the node, and set the socket
            self.node = pipeline.create(node_type)
            self.node.setBoardSocket(socket)

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
                res = getClosesResolution(sensor, sensor_type, width=1300)
                self.node.setResolution(res)
                scale = getClosestIspScale(self.node.getIspSize(), width=1300, videoEncoder=(self.encoder is not None))
                self.node.setIspScale(*scale)

            curr_size = self.node.getVideoSize()
            closest = getClosestVideoSize(*curr_size)
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
            self._control_xlink_in.setMaxDataSize(1) # CameraControl message doesn't use any additional data (only metadata)

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

    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def config_camera(self,
                      # preview: Union[None, str, Tuple[int, int]] = None,
                      size: Union[None, Tuple[int, int], str] = None,
                      resize_mode: ResizeMode = ResizeMode.CROP,
                      fps: Optional[float] = None,
                      resolution: Optional[Union[
                          str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                      ]] = None
                      ) -> None:
        """
        Configure resolution, scale, FPS, etc.
        """

        # TODO
        if fps: self.set_fps(fps)
        if resolution: self._set_resolution(resolution)

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
        :param auto_white_balance: auto white balance to the object
        """

        if not auto_focus and not auto_exposure:
            logging.error(
                'Attempted to control camera with NN, but both Auto-Focus and Auto-Exposure were disabled! Attempt ignored.'
            )
            return
        if 'NNComponent' not in str(type(detection_component)):
            raise ValueError('nn_component must be an instance of NNComponent!')
        if not detection_component._is_detector():
            raise ValueError('nn_component must be a object detection model (YOLO/MobileNetSSD based)!')

        from depthai_sdk.components.control_camera_with_nn import control_camera_with_nn

        control_camera_with_nn(
            pipeline=self._pipeline,
            camera_control=self.node.inputControl,
            nn_output=detection_component.node.out,
            resize_mode=detection_component._ar_resize_mode,
            resolution=self.node.getResolution(),
            nn_size = detection_component._size,
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
            logging.info(
                'Attempted to configure ColorCamera, but this component doesn\'t have it. Config attempt ignored.'
            )
            return

        if self.is_replay():
            logging.info('Tried configuring ColorCamera, but replaying is enabled. Config attempt ignored.')
            return

        self.node: dai.node.ColorCamera

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
            self.node.setResolution(parse_resolution(type(self.node), resolution))
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

    def get_stream_xout(self) -> StreamXout:
        if self.is_replay():
            return ReplayStream(self._source)
        elif self.is_mono():
            return StreamXout(self.node.id, self.stream, name=self.name)
        else:  # ColorCamera
            self.node.setVideoNumFramesPool(self._num_frames_pool)
            self.node.setPreviewNumFramesPool(self._preview_num_frames_pool)
            # node.video instead of preview (self.stream) was used to reduce bandwidth
            # consumption by 2 (3bytes/pixel vs 1.5bytes/pixel)
            return StreamXout(self.node.id, self.node.video, name=self.name)

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

    """
    Available outputs (to the host) of this component
    """

    class Out:
        def __init__(self, camera_component: 'CameraComponent'):
            self._comp = camera_component

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Default output. Uses either camera(), replay(), or encoded() depending on the component settings.
            """
            if self._comp.encoder:
                return self.encoded(pipeline, device)
            elif self._comp.is_replay():
                return self.replay(pipeline, device)
            else:
                return self.camera(pipeline, device)

        def camera(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutFrames:
            """
            Streams camera output to the OAK camera. Produces FramePacket.
            """
            out = XoutFrames(self._comp.get_stream_xout(), self._comp.get_fps())
            out.name = self._comp._source
            return self._comp._create_xout(pipeline, out)

        def replay(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            If depthai-recording was used, it won't stream anything, but it will instead use frames that were sent to the OAK.
            Produces FramePacket.
            """
            out = XoutFrames(ReplayStream(self._comp._source), self._comp.get_fps())
            return self._comp._create_xout(pipeline, out)

        def encoded(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            If encoding was enabled, it will stream bitstream from VideoEncoder node to the host.
            Produces FramePacket.
            """
            if self._comp._encoder_profile == dai.VideoEncoderProperties.Profile.MJPEG:
                out = XoutMjpeg(
                    frames=StreamXout(self._comp.encoder.id, self._comp.encoder.bitstream, name=self._comp.name),
                    color=self._comp.is_color(),
                    lossless=self._comp.encoder.getLossless(),
                    fps=self._comp.encoder.getFrameRate(),
                    frame_shape=self._comp.stream_size
                )
            else:
                out = XoutH26x(
                    frames=StreamXout(self._comp.encoder.id, self._comp.encoder.bitstream, name=self._comp.name),
                    color=self._comp.is_color(),
                    profile=self._comp._encoder_profile,
                    fps=self._comp.encoder.getFrameRate(),
                    frame_shape=self._comp.stream_size
                )
            out.name = self._comp._source
            return self._comp._create_xout(pipeline, out)

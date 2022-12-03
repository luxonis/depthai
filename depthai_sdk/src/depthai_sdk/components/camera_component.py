from typing import Dict

from depthai_sdk.components.camera_helper import *
from depthai_sdk.components.component import Component
from depthai_sdk.components.parser import parse_resolution, parse_encode
from depthai_sdk.oak_outputs.xout import XoutFrames, XoutMjpeg, XoutH26x
from depthai_sdk.oak_outputs.xout_base import XoutBase, StreamXout, ReplayStream
from depthai_sdk.replay import Replay


class CameraComponent(Component):
    # Users should have access to these nodes
    node: Union[dai.node.MonoCamera, dai.node.XLinkIn, dai.node.ColorCamera]
    encoder: dai.node.VideoEncoder

    stream: dai.Node.Output  # Node output to be used as eg. an input into NN
    stream_size: Tuple[int, int]  # Output size

    # Setting passed at init
    _replay: Replay  # Replay module
    _args: Dict
    _control: bool
    _source: str

    # Parsed from settings
    _encoderProfile: dai.VideoEncoderProperties.Profile = None

    def __init__(self,
                 pipeline: dai.Pipeline,
                 source: str,
                 resolution: Union[
                     None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None,
                 fps: Optional[float] = None,
                 encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                 rotation: Optional[int] = None,
                 replay: Optional[Replay] = None,
                 args: Dict = None,
                 ):
        """
        Creates Camera component. This abstracts ColorCamera/MonoCamera nodes and supports mocking the camera when
        recording is passed during OakCamera initialization. Mocking the camera will send frames from the host to the
        OAK device (via XLinkIn node).
        Args:
            source (str): Source of the camera. Either color/rgb/right/left
            resolution (optional): Camera resolution, eg. '800p' or '4k'
            fps (float, optional): Camera FPS
            out (bool, default False): Whether we want to stream frames to the host computer
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            replay (Replay object): Replay
            args (Dict): Use user defined arguments when constructing the pipeline
        """
        super().__init__()
        self.out = self.Out(self)

        # Save passed settings
        self._source = source
        self._replay: Replay = replay
        self._args = args

        if rotation not in [None, 0, 90, 180, 270]:
            raise ValueError(f'Angle {rotation} not supported! Use 0, 90, 180, 270.')

        self._rotation = rotation

        self._create_node(pipeline, source.upper())

        self.encoder = None
        if encode:
            self.encoder = pipeline.createVideoEncoder()
            # MJPEG by default
            self._encoderProfile = parse_encode(encode)

        self._resolution_forced: bool = resolution is not None
        if resolution:
            self._set_resolution(resolution)
        if fps:
            self._set_fps(fps)

    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        if self.is_replay():  # If replay, don't create camera node
            res = self._replay.getShape(self._source)
            # print('resolution', res)
            # resize = getResize(res, width=1200)
            # self._replay.setResizeColor(resize)
            self.node: dai.node.XLinkIn = getattr(self._replay, self._source)
            # print('resize', resize)
            self.node.setMaxDataSize(res[0] * res[1] * 3)
            self.stream_size = res
            self.stream = self.node.out
            if self._rotation:
                rot_manip = self._create_rotation_manip(pipeline) if self._rotation else None
                self.node.out.link(rot_manip.inputImage)
                self.stream = rot_manip.out
            else:
                self.stream = self.node.out

        if isinstance(self.node, dai.node.ColorCamera):
            # DepthAI uses CHW (Planar) channel layout convention for NN inferencing
            self.node.setInterleaved(False)
            # DepthAI uses BGR color order convention for NN inferencing
            self.node.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            self.node.setPreviewNumFramesPool(4)

            cams = device.getCameraSensorNames()
            # print('Available sensors on OAK:', cams)
            sensor_name = cams[dai.CameraBoardSocket.RGB]

            if not self._resolution_forced:  # Find the closest resolution
                self.node.setResolution(getClosesResolution(sensor_name, width=1200))
                scale = getClosestIspScale(self.node.getIspSize(), width=1200, videoEncoder=(self.encoder is not None))
                self.node.setIspScale(*scale)

            self.node.setVideoSize(*getClosestVideoSize(*self.node.getIspSize()))
            self.node.setVideoNumFramesPool(2)  # We will increase it later if we are streaming to host

            self.node.setPreviewSize(*self.node.getVideoSize())
            self.stream_size = self.node.getPreviewSize()
            self.stream = self.node.preview if self.encoder is None else self.node.video

        elif isinstance(self.node, dai.node.MonoCamera):
            self.stream_size = self.node.getResolutionSize()
            self.stream = self.node.out

        if self._args:
            self._config_camera_args(self._args)

        if self._rotation:
            rot_manip = self._create_rotation_manip(pipeline) if self._rotation else None
            self.stream.link(rot_manip.inputImage)
            self.stream = rot_manip.out

        if self.encoder:
            self.encoder.setDefaultProfilePreset(self._get_fps(), self._encoderProfile)
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
                self.stream.link(self.encoder.input)
            else:
                raise ValueError('CameraComponent is neither Color, Mono, nor Replay!')

    def _create_rotation_manip(self, pipeline: dai.Pipeline):
        rot_manip = pipeline.createImageManip()
        rgb_rr = dai.RotatedRect()
        w, h = self.stream_size
        rgb_rr.center.x, rgb_rr.center.y = w // 2, h // 2
        rgb_rr.size.width, rgb_rr.size.height = (w, h) if self._rotation % 180 == 0 else (h, w)
        rgb_rr.angle = self._rotation
        rot_manip.initialConfig.setCropRotatedRect(rgb_rr, False)
        rot_manip.setMaxOutputFrameSize(w * h * 3)
        return rot_manip

    def _create_node(self, pipeline: dai.Pipeline, source: str) -> None:
        """
        Called from __init__ to parse passed `source` argument.
        @param source: User-input source
        """
        if source == "COLOR" or source == "RGB":
            if self.is_replay():
                if 'color' not in self._replay.getStreams():
                    raise Exception('Color stream was not found in specified depthai-recording!')
            else:
                self.node = pipeline.createColorCamera()
                self.node.setBoardSocket(dai.CameraBoardSocket.RGB)

        elif source == "RIGHT" or source == "MONO":
            if self.is_replay():
                if 'right' not in self._replay.getStreams():
                    raise Exception('Right stream was not found in specified depthai-recording!')
            else:
                self.node = pipeline.createMonoCamera()
                self.node.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        elif source == "LEFT":
            if self.is_replay():
                if 'left' not in self._replay.getStreams():
                    raise Exception('Left stream was not found in specified depthai-recording!')
            else:
                self.node = pipeline.createMonoCamera()
                self.node.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            raise ValueError(f"Source name '{source}' not supported!")

    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def config_camera(self,
                      preview: Union[None, str, Tuple[int, int]] = None,
                      fps: Optional[float] = None,
                      resolution: Union[
                          None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None
                      ) -> None:
        """
        Configure resolution, scale, FPS, etc.
        """

        # TODO
        if fps: self._set_fps(fps)
        if resolution: self._set_resolution(resolution)

        if preview:
            from .parser import parse_size
            preview = parse_size(preview)

            self.stream_size = preview

            if self._replay:
                self._replay.setResizeColor(preview)
            elif self.is_color():
                self.node.setPreviewSize(preview)
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
                manulFocus=args.get('manualFocus', None),
                afMode=args.get('afMode', None),
                awbMode=args.get('awbMode', None),
                sceneMode=args.get('sceneMode', None),
                antiBandingMode=args.get('antiBandingMode', None),
                effectMode=args.get('effectMode', None),
                ispScale=args.get('ispScale', None),
                sharpness=args.get('sharpness', None),
                lumaDenoise=args.get('lumaDenoise', None),
                chromaDenoise=args.get('chromaDenoise', None),
            )
        elif self.is_mono():
            self.config_camera(
                fps=args.get('monoFps', None),
                resolution=args.get('monoResolution', None),
            )
        else:  # Replay
            self.config_camera(fps=args.get('fps', None))

    def config_color_camera(self,
                            interleaved: Optional[bool] = None,
                            colorOrder: Union[None, dai.ColorCameraProperties.ColorOrder, str] = None,
                            # Cam control
                            manulFocus: Optional[int] = None,
                            afMode: Optional[dai.CameraControl.AutoFocusMode] = None,
                            awbMode: Optional[dai.CameraControl.AutoWhiteBalanceMode] = None,
                            sceneMode: Optional[dai.CameraControl.SceneMode] = None,
                            antiBandingMode: Optional[dai.CameraControl.AntiBandingMode] = None,
                            effectMode: Optional[dai.CameraControl.EffectMode] = None,
                            # IQ settings
                            ispScale: Optional[Tuple[int, int]] = None,
                            sharpness: Optional[int] = None,
                            lumaDenoise: Optional[int] = None,
                            chromaDenoise: Optional[int] = None,
                            ) -> None:
        if not self.is_color():
            print("Attempted to configure ColorCamera, but this component doesn't have it. Config attempt ignored.")
            return

        if self._replay is not None:
            print('Tried configuring ColorCamera, but replaying is enabled. Config attempt ignored.')
            return

        self.node: dai.node.ColorCamera

        if interleaved: self.node.setInterleaved(interleaved)
        if colorOrder:
            if isinstance(colorOrder, str):
                colorOrder = getattr(dai.ColorCameraProperties.ColorOrder, colorOrder.upper())
            self.node.setColorOrder(colorOrder)

        if manulFocus: self.node.initialControl.setManualFocus(manulFocus)
        if afMode: self.node.initialControl.setAutoFocusMode(afMode)
        if awbMode: self.node.initialControl.setAutoWhiteBalanceMode(awbMode)
        if sceneMode: self.node.initialControl.setSceneMode(sceneMode)
        if antiBandingMode: self.node.initialControl.setAntiBandingMode(antiBandingMode)
        if effectMode: self.node.initialControl.setEffectMode(effectMode)
        # EQ settings
        if ispScale:
            self._resolution_forced = True
            self.node.setIspScale(*ispScale)
        if sharpness: self.node.initialControl.setSharpness(sharpness)
        if lumaDenoise: self.node.initialControl.setLumaDenoise(lumaDenoise)
        if chromaDenoise: self.node.initialControl.setChromaDenoise(chromaDenoise)

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

    def _get_fps(self):
        return (self._replay if self.is_replay() else self.node).getFps()

    def _set_fps(self, fps: float):
        (self._replay if self._replay else self.node).setFps(fps)

    def config_encoder_h26x(self,
                            rate_control_mode: Optional[dai.VideoEncoderProperties.RateControlMode] = None,
                            keyframe_freq: Optional[int] = None,
                            bitrate_kbps: Optional[int] = None,
                            num_b_frames: Optional[int] = None,
                            ):
        if self.encoder is None:
            raise Exception('Video encoder was not enabled!')
        if self._encoderProfile == dai.VideoEncoderProperties.Profile.MJPEG:
            raise Exception('Video encoder was set to MJPEG while trying to configure H26X attributes!')

        if rate_control_mode:
            self.encoder.setRateControlMode(rate_control_mode)
        if keyframe_freq:
            self.encoder.setKeyframeFrequency(keyframe_freq)
        if bitrate_kbps:
            self.encoder.setBitrateKbps(bitrate_kbps)
        if num_b_frames:
            self.encoder.setNumBFrames(num_b_frames)

    def config_encoder_mjpeg(self,
                             quality: Optional[int] = None,
                             lossless: bool = False
                             ):
        if self.encoder is None:
            raise Exception('Video encoder was not enabled!')
        if self._encoderProfile != dai.VideoEncoderProperties.Profile.MJPEG:
            raise Exception(
                f'Video encoder was set to {self._encoderProfile} while trying to configure MJPEG attributes!')

        if quality:
            self.encoder.setQuality(quality)
        if lossless:
            self.encoder.setLossless(lossless)

    def get_stream_xout(self) -> StreamXout:
        if self.is_replay():
            return ReplayStream(self._source)
        elif self.is_mono():
            return StreamXout(self.node.id, self.stream)
        else:  # ColorCamera
            self.node.setVideoNumFramesPool(10)
            return StreamXout(self.node.id, self.stream)

    """
    Available outputs (to the host) of this component
    """

    class Out:
        _comp: 'CameraComponent'

        def __init__(self, cameraComponent: 'CameraComponent'):
            self._comp = cameraComponent

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
            out = XoutFrames(self._comp.get_stream_xout(), self._comp._get_fps())
            out.name = self._comp._source
            return self._comp._create_xout(pipeline, out)

        def replay(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            If depthai-recording was used, it won't stream anything, but it will instead use frames that were sent to the OAK. Produces FramePacket.
            """
            out = XoutFrames(ReplayStream(self._comp._source), self._comp._get_fps())
            return self._comp._create_xout(pipeline, out)

        def encoded(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            If encoding was enabled, it will stream bitstream from VideoEncoder node to the host. Produces FramePacket.
            """
            if self._comp._encoderProfile == dai.VideoEncoderProperties.Profile.MJPEG:
                out = XoutMjpeg(
                    StreamXout(self._comp.encoder.id, self._comp.encoder.bitstream),
                    self._comp.is_color(),
                    self._comp.encoder.getLossless(),
                    self._comp.encoder.getFrameRate()
                )
            else:
                out = XoutH26x(
                    StreamXout(self._comp.encoder.id, self._comp.encoder.bitstream),
                    self._comp.is_color(),
                    self._comp._encoderProfile,
                    self._comp.encoder.getFrameRate()
                )
            out.name = self._comp._source
            return self._comp._create_xout(pipeline, out)

    out: Out

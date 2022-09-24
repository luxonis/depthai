from .component import Component
from typing import Optional, Union, Tuple, Any, Dict
import depthai as dai
from ..replay import Replay
from .camera_helper import *
from .parser import parseResolution, parseEncode
from ..oak_outputs.xout_base import XoutBase, StreamXout, ReplayStream
from ..oak_outputs.xout import XoutFrames


class CameraComponent(Component):
    # Users should have access to these nodes
    node: Union[dai.node.ColorCamera, dai.node.MonoCamera, dai.node.XLinkIn] = None
    encoder: dai.node.VideoEncoder = None

    out: dai.Node.Output = None  # Node output to be used as eg. an input into NN
    out_size: Tuple[int, int]  # Output size

    # Setting passed at init
    _replay: Replay  # Replay module
    _fps: Optional[float]
    _args: Dict
    _control: bool
    _source: str

    # Parsed from settings
    _resolution: Union[None, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None
    _encoderProfile: dai.VideoEncoderProperties.Profile = None

    def __init__(self,
                 pipeline: dai.Pipeline,
                 source: str,
                 resolution: Union[
                     None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None,
                 fps: Optional[float] = None,
                 encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                 control: bool = False,
                 replay: Optional[Replay] = None,
                 args: Any = None,
                 ):
        """
        Args:
            source (str): Source of the camera. Either color/rgb/right/left
            resolution: Camera resolution
            fps: Camera FPS
            out (bool, default False): Whether we want to stream frames to the host computer
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            control (bool, default False): control the camera from the host keyboard (via cv2.waitKey())
            replay (Replay object, optional): Replay
            args (Any, optional): Set the camera components based on user arguments
        """
        super().__init__()

        # Save passed settings
        self._source = source
        self._replay: Replay = replay
        self._fps = fps
        self._args = args
        self._control = control

        self._create_node(pipeline, source.upper())
        if encode:
            self.encoder = pipeline.createVideoEncoder()
            # MJPEG by default
            self._encoderProfile = parseEncode(encode)

        self._resolution = parseResolution(type(self.node), resolution)


    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        # Create the camera
        self._createCam(pipeline, device)

        if self._fps:
            (self._replay if self._replay else self.node).setFps(self._fps)
            # if self.camera:
            #     self.camera.setFps(self._fps)
            # elif self._replay:
            #     self._replay.setFps(self._fps)

    def _createCam(self, pipeline: dai.Pipeline, device: dai.Device) -> None:
        """
        Creates depthai camera node after we connect to OAK camera
        @param pipeline: dai.Pipeline
        """
        if self.isReplay():  # If replay, don't create camera node
            res = self._replay.getShape(self._source)
            resize = getResize(res, width=1200)
            self._replay.setResizeColor(resize)
            self.node: dai.node.XLinkIn = getattr(self._replay, self._source)
            self.node.setMaxDataSize(resize[0] * resize[1] * 3)
            self.out_size = resize
            self.out = self.node.out
            return

        if self._resolution:  # Set sensor resolution as specified by user
            self.node.setResolution(self._resolution)

        if isinstance(self.node, dai.node.ColorCamera):
            # DepthAI uses CHW (Planar) channel layout convention for NN inferencing
            self.node.setInterleaved(False)
            # DepthAI uses BGR color order convention for NN inferencing
            self.node.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            self.node.setPreviewNumFramesPool(4)

            cams = device.getCameraSensorNames()
            # print('Available sensors on OAK:', cams)
            sensorName = cams[dai.CameraBoardSocket.RGB]

            if self._resolution is None:  # Find the closest resolution
                self.node.setResolution(getClosesResolution(sensorName, width=1200))

            scale = getClosestIspScale(self.node.getIspSize(), width=1200)
            self.node.setIspScale(*scale)
            self.node.setPreviewSize(*self.node.getIspSize())
            self.out_size = self.node.getPreviewSize()
            self.out = self.node.preview

        elif isinstance(self.node, dai.node.MonoCamera):
            self.out_size = self.node.getResolutionSize()
            self.out = self.node.out

    def _create_node(self, pipeline: dai.Pipeline, source: str) -> None:
        """
        Called from __init__ to parse passed `source` argument.
        @param source: User-input source
        """
        if source == "COLOR" or source == "RGB":
            if self.isReplay():
                if 'color' not in self._replay.getStreams():
                    raise Exception('Color stream was not found in specified depthai-recording!')
            else:
                self.node = pipeline.createColorCamera()
                self.node.setBoardSocket(dai.CameraBoardSocket.RGB)

        elif source == "RIGHT" or source == "MONO":
            if self.isReplay():
                if 'right' not in self._replay.getStreams():
                    raise Exception('Right stream was not found in specified depthai-recording!')
            else:
                self.node = pipeline.createMonoCamera()
                self.node.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        elif source == "LEFT":
            if self.isReplay():
                if 'left' not in self._replay.getStreams():
                    raise Exception('Left stream was not found in specified depthai-recording!')
            else:
                self.node = pipeline.createMonoCamera()
                self.node.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            raise ValueError(f"Source name '{source}' not supported!")

    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def configure_camera(self,
                         preview: Union[None, str, Tuple[int, int]] = None,  # Set preview size
                         fps: Optional[float] = None  # Set fps
                         ) -> None:
        """
        Configure resolution, scale, FPS, etc.
        """
        return
        # TODO
        if fps:
            if self._replay:
                print(
                    "Setting FPS for depthai-recording isn't yet supported. This configuration attempt will be ignored.")
            else:
                self.node.setFps(fps)

        if preview:
            from .parser import parseSize
            preview = parseSize(preview)

            if self._replay:
                self._replay.setResizeColor(preview)
            elif self.isColor():
                self.node.setPreviewSize(preview)
            else:
                # TODO: Use ImageManip to set mono frame size
                raise NotImplementedError("Not yet implemented")

    def config_color_camera(self,
                            interleaved: Optional[bool] = None,
                            colorOrder: Union[None, dai.ColorCameraProperties.ColorOrder, str] = None,
                            ) -> None:
        raise NotImplementedError()
        # TODO: add option for other configs
        if not self.isColor():
            print("Attempted to configure ColorCamera, but this component doesn't have it. Config attempt ignored.")
            return

        if interleaved is not None:
            self.node.setInterleaved(interleaved)
        if colorOrder:
            if isinstance(colorOrder, str):
                colorOrder = getattr(dai.ColorCameraProperties.ColorOrder, colorOrder.upper())
            self.node.getColorOrder(colorOrder)

    def isReplay(self) -> bool:
        return self._replay is not None

    def isColor(self) -> bool:
        return isinstance(self.node, dai.node.ColorCamera)

    def isMono(self) -> bool:
        return isinstance(self.node, dai.node.MonoCamera)

    def getFps(self):
        return (self._replay if self.isReplay() else self.node).getFps()

    def configure_encoder_h26x(self,
                       rate_control_mode: Optional[dai.VideoEncoderProperties.RateControlMode] = None,
                       keyframe_freq: Optional[int ]= None,
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

    def configure_encoder_mjpeg(self,
                                quality: Optional[int] = None,
                                lossless: bool = False
                                ):
        if self.encoder is None:
            raise Exception('Video encoder was not enabled!')
        if self._encoderProfile != dai.VideoEncoderProperties.Profile.MJPEG:
            raise Exception(f'Video encoder was set to {self._encoderProfile} while trying to configure MJPEG attributes!')

        if quality:
            self.encoder.setQuality(quality)
        if lossless:
            self.encoder.setLossless(lossless)

    def get_stream_xout(self) -> StreamXout:
        # Used by NnComponent
        return ReplayStream(self._source) if  self.isReplay() else StreamXout(self.node.id, self.out)


    """
    Available outputs (to the host) of this component
    """
    def out(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        if self.encoder:
            return self.out_encoded(pipeline, device)
        elif self.isReplay():
            return self.out_replay(pipeline, device)
        else:
            return self.out_camera(pipeline, device)
    def out_camera(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        out = XoutFrames(StreamXout(self.node.id, self.out))
        return super()._create_xout(pipeline, out)

    def out_replay(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        out = XoutFrames(ReplayStream(self._source))
        return super()._create_xout(pipeline, out)

    def out_encoded(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        self.encoder = pipeline.createVideoEncoder()
        self.encoder.setDefaultProfilePreset(self.getFps(), self._encoderProfile)

        if self.isReplay():
            # Create ImageManip to convert to NV12
            type_manip = pipeline.createImageManip()
            type_manip.setFrameType(dai.ImgFrame.Type.NV12)
            type_manip.setMaxOutputFrameSize(self.out_size[0] * self.out_size[1] * 3)
            self.out.link(type_manip.inputImage)
            type_manip.out.link(self.encoder.input)
        elif self.isMono():
            self.node.out.link(self.encoder.input)
        elif self.isColor():
            self.node.video.link(self.encoder.input)
        else:
            raise ValueError('CameraComponent is neither Color, Mono, nor Replay!')

        out = XoutFrames(StreamXout(self.encoder.id, self.encoder.bitstream))
        return super()._create_xout(pipeline, out)

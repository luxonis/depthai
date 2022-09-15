from .component import Component
from typing import Optional, Union, Tuple, Any, Dict
import depthai as dai
from ..replay import Replay
from .camera_helper import *
from .parser import parseResolution, parseEncode


class CameraComponent(Component):
    # Users should have access to these nodes
    camera: Union[dai.node.ColorCamera, dai.node.MonoCamera] = None
    encoder: dai.node.VideoEncoder = None

    out: dai.Node.Output = None  # Node output to be used as eg. an input into NN
    out_size: Tuple[int,int] # Output size

    # Setting passed at init
    _replay: Replay  # Replay module
    _out: Union[None, bool, str]
    _fps: Optional[float]
    _args: Dict
    _control: bool

    # Parsed from settings
    _cam_type: Union[Type[dai.node.ColorCamera], Type[dai.node.MonoCamera]] = None
    _boardSocket: dai.CameraBoardSocket
    _resolution: Union[
        None, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None
    _encoderProfile: dai.VideoEncoderProperties.Profile = None

    def __init__(self,
                 source: str,
                 resolution: Union[
                     None, str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None,
                 fps: Optional[float] = None,
                 out: Union[None, bool, str] = None,
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
        self._replay = replay
        self._out = out
        self._fps = fps
        self._args = args
        self._control = control

        self._parseSource(source.upper())  # Parses out and _cam_type, _boardSocket if replay is passed
        self._encoderProfile = parseEncode(encode)  # Parse encoder profile
        self._resolution = parseResolution(self._cam_type, resolution)

    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        # Create the camera
        self._createCam(pipeline, device)

        # Create XLinkOut after we know camera specs
        if self._out:
            super()._create_xout(
                pipeline,
                type(self._replay) if self._replay else type(self),
                name=self._out,
                out=self.out,
                depthaiMsg=dai.ImgFrame
            )

        if self._fps:
            (self.camera if self.camera else self._replay).setFps(self._fps)
            # if self.camera:
            #     self.camera.setFps(self._fps)
            # elif self._replay:
            #     self._replay.setFps(self._fps)

    def _createCam(self, pipeline: dai.Pipeline, device: dai.Device) -> None:
        """
        Creates depthai camera node after we connect to OAK camera
        @param pipeline: dai.Pipeline
        """
        if self._replay:  # If replay, don't create camera node
            res = self._replay.getShape(self._out)
            resize = getResize(res, width=1200)
            self._replay.setResizeColor(resize)
            xin: dai.node.XLinkIn = getattr(self._replay, self._out)
            xin.setMaxDataSize(resize[0] * resize[1] * 3)
            self.out_size = resize
            self.out = xin.out
            return

        self.camera = pipeline.create(self._cam_type)
        self.camera.setBoardSocket(self._boardSocket)
        if self._resolution:  # Set sensor resolution as specified by user
            self.camera.setResolution(self._resolution)

        if self._cam_type == dai.node.ColorCamera:
            # DepthAI uses CHW (Planar) channel layout convention for NN inferencing
            self.camera.setInterleaved(False)
            # DepthAI uses BGR color order convention for NN inferencing
            self.camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            self.camera.setPreviewNumFramesPool(4)

            cams = device.getCameraSensorNames()
            # print('Available sensors on OAK:', cams)
            sensorName = cams[dai.CameraBoardSocket.RGB]


            if self._resolution is None:  # Find the closest resolution
                self.camera.setResolution(getClosesResolution(sensorName, width=1200))

            scale = getClosestIspScale(self.camera.getIspSize(), width=1200)
            self.camera.setIspScale(*scale)
            self.camera.setPreviewSize(*self.camera.getIspSize())
            self.out_size = self.camera.getPreviewSize()
            self.out = self.camera.preview

        elif self._cam_type == dai.node.MonoCamera:
            self.out_size = self.camera.getResolutionSize()
            self.out = self.camera.out

    def _parseSource(self, source: str) -> None:
        """
        Called from __init__ to parse passed `source` argument.
        @param source: User-input source
        """
        if source == "COLOR" or source == "RGB":
            if self._replay:
                if 'color' not in self._replay.getStreams():
                    raise Exception('Color stream was not found in specified depthai-recording!')
            else:
                self._cam_type = dai.node.ColorCamera
                self._boardSocket = dai.CameraBoardSocket.RGB

        elif source == "RIGHT" or source == "MONO":
            if self._replay:
                if 'right' not in self._replay.getStreams():
                    raise Exception('Right stream was not found in specified depthai-recording!')
            else:
                self._cam_type = dai.node.MonoCamera
                self._boardSocket = dai.CameraBoardSocket.RIGHT

        elif source == "LEFT":
            if self._replay:
                if 'left' not in self._replay.getStreams():
                    raise Exception('Left stream was not found in specified depthai-recording!')
            else:
                self._cam_type = dai.node.MonoCamera
                self._boardSocket = dai.CameraBoardSocket.LEFT
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
                self.camera.setFps(fps)

        if preview:
            from .parser import parseSize
            preview = parseSize(preview)

            if self._replay:
                self._replay.setResizeColor(preview)
            elif self._isColor():
                self.camera.setPreviewSize(preview)
            else:
                # TODO: Use ImageManip to set mono frame size
                raise NotImplementedError("Not yet implemented")

    def config_color_camera(self,
                            interleaved: Optional[bool] = None,
                            colorOrder: Union[None, dai.ColorCameraProperties.ColorOrder, str] = None,
                            ) -> None:
        raise NotImplementedError()
        # TODO: add option for other configs
        if not self._isColor():
            print("Attempted to configure ColorCamera, but this component doesn't have it. Config attempt ignored.")
            return

        if interleaved is not None:
            self.camera.setInterleaved(interleaved)
        if colorOrder:
            if isinstance(colorOrder, str):
                colorOrder = getattr(dai.ColorCameraProperties.ColorOrder, colorOrder.upper())
            self.camera.getColorOrder(colorOrder)

    def _isColor(self) -> bool:
        return self._cam_type == dai.node.ColorCamera

    def _isMono(self) -> bool:
        return self._cam_type == dai.node.MonoCamera

    def configure_encoder(self,
                          ):
        """
        Configure quality, enable lossless,
        """
        if self.encoder is None:
            print('Video encoder was not enabled! This configuration attempt will be ignored.')
            return

        raise NotImplementedError()

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

    # Setting passed at init
    _replay: Replay  # Replay module
    _out: Union[None, bool, str]
    _fps: Optional[float]
    _args: Dict
    _control: bool

    # Parsed from settings
    _camType: Type
    _boardSocket: dai.CameraBoardSocket
    _resolution: Union[None, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution] = None
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

        self._parseSource(source.upper())  # Parses out and _camType, _boardSocket if replay is passed
        self._encoderProfile = parseEncode(encode)  # Parse encoder profile
        self._resolution = parseResolution(self._camType, resolution)

    def updateDeviceInfo(self, pipeline: dai.Pipeline, device: dai.Device):
        # Create the camera
        self._createCam(pipeline, device)

        if self._replay:
            return  # No need to config anything

        # Create XLinkOut after we know camera specs
        if self._out:
            super().createXOut(
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
            return

        self.camera = pipeline.create(self._camType)
        self.camera.setBoardSocket(self._boardSocket)

        if self._camType == dai.node.ColorCamera:
            # DepthAI uses CHW (Planar) channel layout convention for NN inferencing
            self.camera.setInterleaved(False)
            # DepthAI uses BGR color order convention for NN inferencing
            self.camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            self.camera.setPreviewNumFramesPool(4)

            # Use full FOV, ISP downscale to get about 1300 pixel width (for visualization)
            cams = device.getCameraSensorNames()
            print('Available sensors on OAK:', cams)
            sensorName = cams[dai.CameraBoardSocket.RGB]
            # Set max sensor resolution (for full FOV)
            self.camera.setResolution(cameraSensorResolution(sensorName))
            camRes = cameraSensorResolutionSize(sensorName)
            scale = getClosestIspScale(camRes, width=1200)
            self.camera.setIspScale(*scale)

            self.camera.setPreviewSize(
                int(camRes[0] * scale[0]/scale[1]),
                int(camRes[1] * scale[0]/scale[1]),
            )

            self.out = self.camera.preview  # For full FOV

        else:
            self.out = self.camera.out

    def _parseSource(self, source: str) -> None:
        """
        Called from __init__ to parse passed `source` argument.
        @param source: User-input source
        """
        if source == "COLOR" or source == "RGB":
            if self._replay:
                if not self._replay.color:
                    raise Exception('Color stream was not found in specified depthai-recording!')
                self.out = self._replay.color.out
            else:
                self._camType = dai.node.ColorCamera
                self._boardSocket = dai.CameraBoardSocket.RGB

        elif source == "RIGHT" or source == "MONO":
            if self._replay:
                if not self._replay.right:
                    raise Exception('Right stream was not found in specified depthai-recording!')
                self.out = self._replay.right.out
            else:
                self._camType = dai.node.MonoCamera
                self._boardSocket = dai.CameraBoardSocket.RIGHT

        elif source == "LEFT":
            if self._replay:
                if not self._replay.left:
                    raise Exception('Left stream was not found in specified depthai-recording!')
                self.out = self._replay.left.out
            else:
                self._camType = dai.node.MonoCamera
                self._boardSocket = dai.CameraBoardSocket.LEFT
        else:
            raise ValueError(f"Source name '{source}' not supported!")

    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def configureCamera(self,
                        preview: Union[None, str, Tuple[int, int]] = None,  # Set preview size
                        fps: Optional[float] = None  # Set fps
                        ) -> None:
        """
        Configure resolution, scale, FPS, etc.
        """
        raise NotImplementedError()
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

    def configColorCamera(self,
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
        return isinstance(self.camera, dai.node.ColorCamera)

    def _isMono(self) -> bool:
        return isinstance(self.camera, dai.node.MonoCamera)

    def configureEncoder(self,
                         ):
        """
        Configure quality, enable lossless,
        """
        if self.encoder is None:
            print('Video encoder was not enabled! This configuration attempt will be ignored.')
            return

        raise NotImplementedError()

from .component import Component
from typing import Optional, Union, Any
import depthai as dai
from ..replay import Replay

class CameraComponent(Component):
    # Users should have access to these nodes
    color: dai.node.ColorCamera = None
    mono: dai.node.MonoCamera = None
    encoder: dai.node.VideoEncoder = None
    replay: Replay = None # Replay module

    out: dai.Node.Output = None

    def __init__(self,
        pipeline: dai.Pipeline,
        source: Union[str, Replay], #
        name: Optional[str] = None,
        out: bool = False,
        encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
        control: bool = False,
        args: Any = None,
        ):
        """
        Args:
            pipeline (dai.Pipeline)
            source (str or Replay): Source of the camera. Either color/rgb/right/left or Replay object
            name (str, optional): name of the camera
            out (bool, default False): Whether we want to stream frames to the host computer
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            control (bool, default False): control the camera from the host keyboard (via cv2.waitKey())
            args (Any, optional): Set the camera components based on user arguments
        """

        self.pipeline = pipeline
        self._parseSource(source)

    def _parseSource(self, source: Optional[str] = None):
        if source.upper() == "COLOR" or source.upper() == "RGB":
            self.color = self.pipeline.create(dai.node.ColorCamera)
            self.color.setBoardSocket(dai.CameraBoardSocket.RGB)
            self.out = self.color.preview
        elif source.upper() == "RIGHT" or source.upper() == "MONO":
            self.mono = self.pipeline.create(dai.node.MonoCamera)
            self.mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            self.out = self.mono.out
        elif source.upper() == "LEFT":
            self.mono = self.pipeline.create(dai.node.MonoCamera)
            self.mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
            self.out = self.mono.out
        else:
            if self._isUrl(source):
                if self._isYoutubeLink(source):
                    from ..utils import downloadYTVideo
                    # Overwrite source - so Replay class can use it
                    source = str(downloadYTVideo(source))
                else:
                    # TODO: download video/image(s) from the internet
                    raise NotImplementedError("Only YouTube video download is currently supported!")
                
            self.replay = Replay(source, self.pipeline)


    def _isYoutubeLink(self, source: str) -> bool:
        return "youtube.com" in source

    def _isUrl(self, source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")


    def _createXLinkOut(self):
        a = 5

    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def configureCamera(self, 

        ): 
        """
        Configure resolution, scale, FPS, etc.
        """
        a = 5

    def configureEncoder(self,
        ):
        """
        Configure quality, enable lossless,
        """
        if self.encoder is None:
            print('Video encoder was not enabled! This configuration attempt will be ignored.')
            return

        # self.encoer.

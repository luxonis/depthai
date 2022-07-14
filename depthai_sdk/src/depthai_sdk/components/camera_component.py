from .component import Component
from typing import Optional, Union, Any
import depthai as dai
from ..replay import Replay

class CameraComponent(Component):

    _colorcam: dai.node.ColorCamera = None
    _monocam: dai.node.MonoCamera = None
    _replay: Replay = None # Replay module

    _enc: dai.node.VideoEncoder = None

    _source = ""

    def __init__(self,
        pipeline: dai.Pipeline,
        source: Optional[str] = None, #
        name: Optional[str] = None,
        out: bool = False,
        encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
        control: bool = False,
        args: Any = None,
        ):
        """
        Args:
            pipeline (dai.Pipeline)
            source (str, optional): Source of the camera. Either color/rgb/right/left or Recorded stream or YouTube link
            name (str, optional): name of the camera
            out (bool, default False): Whether we want to stream frames to the host computer
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            control (bool, default False): control the camera from the host keyboard (via cv2.waitKey())
            args (Any, optional): Set the camera components based on user arguments
        """

        self.pipeline = pipeline
        self._parseSource(source)

        if out:
            if encode is not None:
                self._enc = pipeline.createVideoEncoder()

    def _parseSource(self, source: Optional[str] = None):
        if source is not None:
            if source.upper() == "COLOR" or source.upper() == "RGB":
                self._colorcam = self.pipeline.create(dai.node.ColorCamera)
                self._source = "color"
            elif source.upper() == "RIGHT" or source.upper() == "MONO":
                self._monocam = self.pipeline.create(dai.node.MonoCamera)
                self._monocam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                self._source = "right"
            elif source.upper() == "LEFT":
                self._monocam = self.pipeline.create(dai.node.MonoCamera)
                self._monocam.setBoardSocket(dai.CameraBoardSocket.LEFT)
                self._source = "left"
            else:
                if self._isUrl(source):
                    if self._isYoutubeLink(source):
                        from ..utils import downloadYTVideo
                        # Overwrite source - so Replay class can use it
                        source = str(downloadYTVideo(source))
                    else:
                        # TODO: download video/image(s) from the internet
                        raise NotImplementedError("Only YouTube video download is currently supported!")
                    
                self._replay = Replay(source, self.pipeline)
                self._source = "replay"

        else:
            self._colorcam = self.pipeline.create(dai.node.ColorCamera)

    def _isYoutubeLink(self, source: str) -> bool:
        return "youtube.com" in source

    def _isUrl(self, source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")

    def _getSource(self):
        if self._source == 'color':
            return self._colorcam
        elif self._source == 'left':


    def _createXLinkOut(self):
        a = 5


    def configureCamera(self, 

        ): 
        """
        Configure resolution, FPS, etc.
        """
        a = 5

    def configureEncoder(self,
        ):
        """
        Configure quality, enable lossless,
        """
        if self._enc is None:
            raise Exception('Video encoder was not enabled!')

        self._enc.set

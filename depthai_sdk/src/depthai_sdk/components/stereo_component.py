from .component import Component
from typing import Optional, Union, Tuple, Any
import depthai as dai
from ..replay import Replay
from .camera_component import CameraComponent

class StereoComponent(Component):
    # Users should have access to these nodes
    node: dai.node.StereoDepth
    _replay: Optional[Replay] = None # Replay module

    out: dai.Node.Output # depth output
    depth: dai.Node.Output
    disparity: dai.Node.Output

    def __init__(self,
        pipeline: dai.Pipeline,
        resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
        fps: Optional[float] = None,
        name: Optional[str] = None,
        out: Optional[str] = None, # 'depth', 'disparity', both seperated by comma? TBD
        left: Union[None, dai.Node.Output, CameraComponent] = None, # Left mono camera
        right: Union[None, dai.Node.Output, CameraComponent] = None, # Right mono camera
        encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
        control: bool = False,
        replay: Optional[Replay] = None,
        args: Any = None,
        ):
        """
        Args:
            pipeline (dai.Pipeline)
            name (str, optional): name of the camera
            out (str, optional): 'depth', 'disparity', both seperated by comma? TBD
            left (None / dai.None.Output / CameraComponent): Left mono camera source. Will get handled by Camera object.
            right (None / dai.None.Output / CameraComponent): Right mono camera source. Will get handled by Camera object.
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            control (bool, default False): control the camera from the host keyboard (via cv2.waitKey())
            replay (Replay object, optional): Replay
            args (Any, optional): Set the camera components based on user arguments
        """

        self.pipeline = pipeline
        self.replay = replay

        if replay:
            print('Replay found, using that')
            if not replay.stereo:
                raise Exception('Stereo stream was not found in specified depthai-recording!')
            self.node = replay.stereo
        else:
            print('creating Left/Right')
            from .camera_component import CameraComponent
            if not left: left = CameraComponent(pipeline, 'left', resolution, fps)
            if not right: right = CameraComponent(pipeline, 'right', resolution, fps)
            # TODO create StereoDepth
            if isinstance(left, CameraComponent): left = left.out
            if isinstance(right, CameraComponent): right = right.out

            self.node = pipeline.createStereoDepth()
            left.link(self.node.left)
            right.link(self.node.right)
        
        self.out = self.node.depth
        self.depth = self.node.depth
        self.disparity = self.node.disparity

    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def configureStereo(self,
        confidence: Optional[int] = None,
        align: Optional[dai.CameraBoardSocket] = None,
        extended: Optional[bool] = None,
        subpixel: Optional[bool] = None,
        lrcheck: Optional[bool] = None,
        ) -> None: 
        """
        Configure StereoDepth modes, filters, etc.
        """
        if confidence: self.node.setConfidenceThreshold(confidence)
        if align: self.node.setDepthAlign(align)
        if extended: self.node.setExtendedDisparity(extended)
        if subpixel: self.node.setExtendedDisparity(subpixel)
        if lrcheck: self.node.setExtendedDisparity(lrcheck)

    def configureEncoder(self,
        ):
        """
        Configure quality, enable lossless,
        """
        if self.encoder is None:
            print('Video encoder was not enabled! This configuration attempt will be ignored.')
            return

        # self.encoer.

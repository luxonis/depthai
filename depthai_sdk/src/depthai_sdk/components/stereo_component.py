from .component import Component
from .camera_component import CameraComponent
from typing import Optional, Union, Tuple, Any
import depthai as dai
from ..replay import Replay


class StereoComponent(Component):
    # Users should have access to these nodes
    node: dai.node.StereoDepth
    _replay: Optional[Replay] = None  # Replay module

    out: dai.Node.Output  # depth output
    depth: dai.Node.Output
    disparity: dai.Node.Output

    left: Union[None, dai.Node.Output, CameraComponent] = None
    right: Union[None, dai.Node.Output, CameraComponent] = None

    def __init__(self,
                 resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
                 fps: Optional[float] = None,
                 out: Optional[str] = None,  # 'depth', 'disparity', both separated by comma? TBD
                 left: Union[None, dai.Node.Output, CameraComponent] = None,  # Left mono camera
                 right: Union[None, dai.Node.Output, CameraComponent] = None,  # Right mono camera
                 encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                 control: bool = False,
                 replay: Optional[Replay] = None,
                 args: Any = None,
                 ):
        """
        Args:
            out (str, optional): 'depth', 'disparity', both seperated by comma? TBD
            left (None / dai.None.Output / CameraComponent): Left mono camera source. Will get handled by Camera object.
            right (None / dai.None.Output / CameraComponent): Right mono camera source. Will get handled by Camera object.
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            control (bool, default False): control the camera from the host keyboard (via cv2.waitKey())
            replay (Replay object, optional): Replay
            args (Any, optional): Set the camera components based on user arguments
        """
        super().__init__()
        self.replay = replay

        if replay:
            print('Replay found, using that')
            if not replay.stereo:
                raise Exception('Stereo stream was not found in specified depthai-recording!')
            self.node = replay.stereo
        else:
            self.left = left
            self.right = right
            from .camera_component import CameraComponent
            if not left:
                left = CameraComponent('left', resolution, fps)
            if not right:
                right = CameraComponent('right', resolution, fps)
            # TODO create StereoDepth
            if isinstance(left, CameraComponent):
                left = left.out
            if isinstance(right, CameraComponent):
                right = right.out

            self.node = pipeline.createStereoDepth()
            left.link(self.node.left)
            right.link(self.node.right)

        self.out = self.node.depth
        self.depth = self.node.depth
        self.disparity = self.node.disparity

    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        # TODO: disable mono cams if OAK doesn't have them
        pass

    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def configure_stereo(self,
                         confidence: Optional[int] = None,
                         align: Optional[dai.CameraBoardSocket] = None,
                         extended: Optional[bool] = None,
                         subpixel: Optional[bool] = None,
                         lr_check: Optional[bool] = None,
                         ) -> None:
        """
        Configure StereoDepth modes, filters, etc.
        """
        initialConfig = dai.StereoDepthConfig()
        if confidence: initialConfig.setConfidenceThreshold(confidence)
        if align: initialConfig.setDepthAlign(align)
        if extended: initialConfig.setExtendedDisparity(extended)
        if subpixel: initialConfig.setSubpixel(subpixel)
        if lr_check: initialConfig.setExtendedDisparity(lr_check)

    def configure_encoder(self,
                          ):
        """
        Configure quality, enable lossless,
        """
        if self.encoder is None:
            print('Video encoder was not enabled! This configuration attempt will be ignored.')
            return

        raise NotImplementedError()

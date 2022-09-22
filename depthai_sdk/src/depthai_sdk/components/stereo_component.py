from .component import Component
from .camera_component import CameraComponent
from typing import Optional, Union, Tuple, Any, Dict, Callable
import depthai as dai

from ..classes.xout_base import XoutBase, StreamXout
from ..classes.xout import XoutDisparity, XoutDepth
from ..replay import Replay
from .parser import parse_cam_socket


class StereoComponent(Component):
    # Users should have access to these nodes
    node: dai.node.StereoDepth

    left: Union[None, dai.Node.Output, CameraComponent] = None
    right: Union[None, dai.Node.Output, CameraComponent] = None

    _replay: Optional[Replay]  # Replay module
    _resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution]
    _fps: Optional[float]
    _control: bool
    _args: Dict
    # Configs
    _align: dai.CameraBoardSocket = None
    _initialConfig: dai.StereoDepthConfig = None

    def __init__(self,
                 resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
                 fps: Optional[float] = None,
                 left: Union[None, dai.Node.Output, CameraComponent] = None,  # Left mono camera
                 right: Union[None, dai.Node.Output, CameraComponent] = None,  # Right mono camera
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
        self._replay = replay
        self._resolution = resolution
        self._fps = fps
        self._args = args
        self._control = control

        self.left = left
        self.right = right

    @property
    def depth(self) -> dai.Node.Output:
        """
        Depth output from the StereoDepth node.
        """
        return self.node.depth

    @property
    def disparity(self) -> dai.Node.Output:
        """
        Disparity output from the StereoDepth node.
        """
        return self.node.disparity

    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        if self._replay:
            print('Replay found, using that')
            if not self._replay.stereo:
                raise Exception('Stereo stream was not found in specified depthai-recording!')
            self.node = self._replay.stereo

        # TODO: check sensor names / device name whether it has stereo camera pair (or maybe calibration?)
        if len(device.getCameraSensorNames()) == 1:
            raise Exception('OAK-1 camera does not have Stereo camera pair!')

        from .camera_component import CameraComponent
        if not self.left:
            self.left = CameraComponent('left', self._resolution, self._fps)
            self.left._update_device_info(pipeline, device, version)
        if not self.right:
            self.right = CameraComponent('right', self._resolution, self._fps)
            self.right._update_device_info(pipeline, device, version)

        self.node = pipeline.createStereoDepth()
        # TODO: use self._args to setup the StereoDepth node

        if isinstance(self.left, CameraComponent):
            self.left = self.left.out
        if isinstance(self.right, CameraComponent):
            self.right = self.right.out

        if self._align:
            self.node.setDepthAlign(self._align)
        # if self._initialConfig:
        #     self.node.initialConfig = self._initialConfig

        # Connect Mono cameras to the StereoDepth node
        self.left.link(self.node.left)
        self.right.link(self.node.right)


    # Should be mono/color camera agnostic. Also call this from __init__ if args is enabled
    def configure_stereo(self,
                         confidence: Optional[int] = None,
                         align: Union[None, str, dai.CameraBoardSocket] = None,
                         extended: Optional[bool] = None,
                         subpixel: Optional[bool] = None,
                         lr_check: Optional[bool] = None,
                         ) -> None:
        """
        Configure StereoDepth modes, filters, etc.
        """
        _initialConfig = dai.StereoDepthConfig()
        if confidence: _initialConfig.setConfidenceThreshold(confidence)
        if align: self._align = parse_cam_socket(align)
        if extended: _initialConfig.setExtendedDisparity(extended)
        if subpixel: _initialConfig.setSubpixel(subpixel)
        if lr_check: _initialConfig.setExtendedDisparity(lr_check)

    """
    Available outputs (to the host) of this component
    """
    def out(self, pipeline: dai.Pipeline, callback: Callable) -> XoutBase:
        # By default, we want to show disparity
        return self.out_disparity(pipeline, callback)
    def out_disparity(self, pipeline: dai.Pipeline, callback: Callable) -> XoutDisparity:
        out = XoutDisparity(callback, StreamXout(self.node.id, self.disparity), self.node.getMaxDisparity())
        super()._create_xout(pipeline, out)
        return out

    def out_depth(self, pipeline: dai.Pipeline, callback: Callable) -> XoutDepth:
        out = XoutDepth(callback, StreamXout(self.node.id, self.depth))
        super()._create_xout(pipeline, out)
        return out

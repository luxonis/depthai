from .component import Component
from .camera_component import CameraComponent
from typing import Optional, Union, Tuple, Any, Dict, Callable
import depthai as dai

from ..oak_outputs.xout_base import XoutBase, StreamXout
from ..oak_outputs.xout import XoutDisparity, XoutDepth
from ..replay import Replay
from .parser import parse_cam_socket, parse_median_filter


class StereoComponent(Component):
    # Users should have access to these nodes
    node: dai.node.StereoDepth = None

    left: Union[None, CameraComponent, dai.node.MonoCamera] = None
    right: Union[None, CameraComponent, dai.node.MonoCamera] = None

    @property
    def depth(self) -> dai.Node.Output:
        # Depth output from the StereoDepth node.
        return self.node.depth

    @property
    def disparity(self) -> dai.Node.Output:
        # Disparity output from the StereoDepth node.
        return self.node.disparity


    _replay: Optional[Replay]  # Replay module
    _resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution]
    _fps: Optional[float]
    _args: Dict

    def __init__(self,
                 pipeline: dai.Pipeline,
                 resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
                 fps: Optional[float] = None,
                 left: Union[None, CameraComponent, dai.node.MonoCamera] = None,  # Left mono camera
                 right: Union[None, CameraComponent, dai.node.MonoCamera] = None,  # Right mono camera
                 replay: Optional[Replay] = None,
                 args: Any = None,
                 ):
        """
        Args:
            out (str, optional): 'depth', 'disparity', both seperated by comma? TBD
            left (None / dai.None.Output / CameraComponent): Left mono camera source. Will get handled by Camera object.
            right (None / dai.None.Output / CameraComponent): Right mono camera source. Will get handled by Camera object.
            encode: Encode streams before sending them to the host. Either True (use default), or mjpeg/h264/h265
            replay (Replay object, optional): Replay
            args (Any, optional): Set the camera components based on user arguments
        """
        super().__init__()
        self._replay = replay
        self._resolution = resolution
        self._fps = fps
        self._args = args

        self.left = left
        self.right = right

        self.node = pipeline.createStereoDepth()
        self.node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        if self._replay:
            print('Replay found, using that')
            if not self._replay.stereo:
                raise Exception('Stereo stream was not found in specified depthai-recording!')
            self.node = self._replay.stereo

        # TODO: check sensor names / device name whether it has stereo camera pair (or maybe calibration?)
        if len(device.getCameraSensorNames()) == 1:
            raise Exception('OAK-1 camera does not have Stereo camera pair!')

        if not self.left:
            self.left = CameraComponent(pipeline, 'left', self._resolution, self._fps)
            self.left._update_device_info(pipeline, device, version)
        if not self.right:
            self.right = CameraComponent(pipeline, 'right', self._resolution, self._fps)
            self.right._update_device_info(pipeline, device, version)

        # TODO: use self._args to setup the StereoDepth node

        if isinstance(self.left, CameraComponent):
            self.left = self.left.node # CameraComponent -> node
        if isinstance(self.right, CameraComponent):
            self.right = self.right.node # CameraComponent -> node

        # Connect Mono cameras to the StereoDepth node
        self.left.out.link(self.node.left)
        self.right.out.link(self.node.right)

        if self._args:
            self._config_stereo_args(self._args)

    def _config_stereo_args(self, args: Dict):
        if not isinstance(args, Dict):
            args = vars(args)  # Namespace -> Dict
        self.config_stereo(
            confidence=args.get('disparityConfidenceThreshold', None),
            median=args.get('stereoMedianSize', None),
            extended=args.get('extendedDisparity', None),
            subpixel=args.get('subpixel', None),
            lrCheck=args.get('lrCheck', None),
            sigma=args.get('sigma', None),
            lrCheckThreshold=args.get('lrcThreshold', None),
        )

    def config_stereo(self,
                      confidence: Optional[int] = None,
                      align: Union[None, str, dai.CameraBoardSocket] = None,
                      median: Union[None, int, dai.MedianFilter] = None,
                      extended: Optional[bool] = None,
                      subpixel: Optional[bool] = None,
                      lrCheck: Optional[bool] = None,
                      sigma: Optional[int] = None,
                      lrCheckThreshold: Optional[int] = None,
                      ) -> None:
        """
        Configures StereoDepth modes and options.
        """
        if confidence: self.node.initialConfig.setConfidenceThreshold(confidence)
        if align: self.node.setDepthAlign(parse_cam_socket(align))
        if median: self.node.setMedianFilter(parse_median_filter(median))
        if extended: self.node.initialConfig.setExtendedDisparity(extended)
        if subpixel: self.node.initialConfig.setSubpixel(subpixel)
        if lrCheck: self.node.initialConfig.setLeftRightCheck(lrCheck)
        if sigma: self.node.initialConfig.setBilateralFilterSigma(sigma)
        if lrCheckThreshold: self.node.initialConfig.setLeftRightCheckThreshold(lrCheckThreshold)


    def _get_disparity_factor(self, device: dai.Device) -> float:
        """
        Calculates the disparity factor used to calculate depth from disparity.
        `depth = disparity_factor / disparity`
        @param device: OAK device
        """
        calib = device.readCalibration()
        baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, self.right.getResolutionSize())
        focalLength = intrinsics[0][0]
        disp_levels = self.node.getMaxDisparity() / 95
        return baseline * focalLength * disp_levels

    """
    Available outputs (to the host) of this component
    """

    def out(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        # By default, we want to show disparity
        return self.out_depth(pipeline, device)

    def out_disparity(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        out = XoutDisparity(StreamXout(self.node.id, self.disparity), self.node.getMaxDisparity(), self.left.getFps())
        return super()._create_xout(pipeline, out)

    def out_depth(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        out = XoutDepth(device, StreamXout(self.node.id, self.depth), self.left.getFps())
        return super()._create_xout(pipeline, out)

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
    node: dai.node.StereoDepth

    left: Union[None, CameraComponent, dai.node.MonoCamera]
    right: Union[None, CameraComponent, dai.node.MonoCamera]

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
            pipeline (dai.Pipeline): DepthAI pipeline
            resolution (str/SensorResolution): If monochrome cameras aren't already passed, create them and set specified resolution
            fps (float): If monochrome cameras aren't already passed, create them and set specified FPS
            left (None / dai.None.Output / CameraComponent): Left mono camera source. Will get handled by Camera object.
            right (None / dai.None.Output / CameraComponent): Right mono camera source. Will get handled by Camera object.
            replay (Replay object, optional): Replay
            args (Any, optional): Use user defined arguments when constructing the pipeline
        """
        super().__init__()
        self.out = self.Out(self)

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
            self._replay.initStereoDepth(self.node)
        else:
            # TODO: check sensor names / device name whether it has stereo camera pair (or maybe calibration?)
            if len(device.getCameraSensorNames()) == 1:
                raise Exception('OAK-1 camera does not have Stereo camera pair!')

            if not self.left:
                self.left = CameraComponent(pipeline, 'left', self._resolution, self._fps, replay=self._replay)
                self.left._update_device_info(pipeline, device, version)
            if not self.right:
                self.right = CameraComponent(pipeline, 'right', self._resolution, self._fps, replay=self._replay)
                self.right._update_device_info(pipeline, device, version)

            # TODO: use self._args to setup the StereoDepth node

            if isinstance(self.left, CameraComponent):
                self.left = self.left.node  # CameraComponent -> node
            if isinstance(self.right, CameraComponent):
                self.right = self.right.node  # CameraComponent -> node

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

    class Out:
        _comp: 'StereoComponent'

        def __init__(self, stereoComponent: 'StereoComponent'):
            self._comp = stereoComponent

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            # By default, we want to show disparity
            return self.depth(pipeline, device)

        def disparity(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            fps = self._comp.left.getFps() if self._comp._replay is None else self._comp._replay.getFps()
            out = XoutDisparity(
                StreamXout(self._comp.node.id, self._comp.disparity),
                self._comp.node.getMaxDisparity(),
                fps
            )
            return self._comp._create_xout(pipeline, out)

        def depth(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            fps = self._comp.left.getFps() if self._comp._replay is None else self._comp._replay.getFps()
            out = XoutDepth(device, StreamXout(self._comp.node.id, self._comp.depth), fps)
            return self._comp._create_xout(pipeline, out)

    out: Out

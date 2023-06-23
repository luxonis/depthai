from types import SimpleNamespace
import depthai as dai

from ..previews import Previews
from typing import Tuple, Optional

class PipelineManager:
    """
    Manager class handling different :obj:`depthai.Pipeline` operations. Most of the functions wrap up nodes creation
    and connection logic onto a set of convenience functions.
    """

    def __init__(self, openvinoVersion=None, poeQuality=100, lowCapabilities=False, lowBandwidth=False):
        self.openvinoVersion=openvinoVersion
        self.poeQuality = poeQuality
        self.lowCapabilities = lowCapabilities
        self.lowBandwidth = lowBandwidth

        #: depthai.Pipeline: Ready to use requested pipeline. Can be passed to :obj:`depthai.Device` to start execution
        self.pipeline = dai.Pipeline()
        #: types.SimpleNamespace: Contains all nodes added to the :attr:`pipeline` object, can be used to conveniently access nodes by their name
        self.nodes = SimpleNamespace()

        if openvinoVersion is not None:
            self.pipeline.setOpenVINOVersion(openvinoVersion)

    #: depthai.OpenVINO.Version: OpenVINO version which will be used in pipeline
    openvinoVersion = None
    #: int, Optional: PoE encoding quality, can decrease frame quality but decrease latency
    poeQuality = None
    #: bool: If set to :code:`True`, manager will MJPEG-encode the packets sent from device to host to lower the bandwidth usage. **Can break** if more than 3 encoded outputs requested
    lowBandwidth = False
    #: bool: If set to :code:`True`, manager will try to optimize the pipeline to reduce the amount of host-side calculations (useful for RPi or other embedded systems)
    lowCapabilities = False

    _depthConfig = dai.StereoDepthConfig()
    _rgbConfig = dai.CameraControl()
    _leftConfig = dai.CameraControl()
    _rightConfig = dai.CameraControl()

    _depthConfigInputQueue = None
    _rgbConfigInputQueue = None
    _leftConfigInputQueue = None
    _rightConfigInputQueue = None

    def setNnManager(self, nnManager):
        """
        Assigns NN manager. It also syncs the pipeline versions between those two objects

        Args:
            nnManager (depthai_sdk.managers.NNetManager): NN manager instance
        """
        self.nnManager = nnManager
        if self.openvinoVersion is None and self.nnManager.openvinoVersion:
            self.pipeline.setOpenVINOVersion(self.nnManager.openvinoVersion)
        else:
            self.nnManager.openvinoVersion = self.pipeline.getOpenVINOVersion()

    def createDefaultQueues(self, device):
        """
        Creates default queues for config updates

        Args:
            device (depthai.Device): Running device instance
        """

        if hasattr(self.nodes, "stereo"):
            self._depthConfigInputQueue = device.getInputQueue("stereoConfig")
        if hasattr(self.nodes, "camRgb"):
            self._rgbConfigInputQueue = device.getInputQueue(Previews.color.name + "_control")
        if hasattr(self.nodes, "monoLeft"):
            self._leftConfigInputQueue = device.getInputQueue(Previews.left.name + "_control")
        if hasattr(self.nodes, "monoRight"):
            self._rightConfigInputQueue = device.getInputQueue(Previews.right.name + "_control")

    def closeDefaultQueues(self):
        """
        Creates default queues for config updates

        Args:
            device (depthai.Device): Running device instance
        """

        if self._depthConfigInputQueue is not None:
            self._depthConfigInputQueue.close()
        if self._rgbConfigInputQueue is not None:
            self._rgbConfigInputQueue.close()
        if self._leftConfigInputQueue is not None:
            self._leftConfigInputQueue.close()
        if self._rightConfigInputQueue is not None:
            self._rightConfigInputQueue.close()

    def __calcEncodeableSize(self, sourceSize):
        w, h = sourceSize
        if w % 16 > 0:
            newW = w - (w % 16)
            h = int((newW / w) * h)
            w = int(newW)
        if h % 2 > 0:
            h -= 1
        return w, h

    def _mjpegLink(self, node, xout, nodeOutput):
        print("Creating MJPEG link for {} node and {} xlink stream...".format(node.getName(), xout.getStreamName()))
        videnc = self.pipeline.createVideoEncoder()
        if isinstance(node, dai.node.ColorCamera):
            if node.video == nodeOutput:
                size = self.__calcEncodeableSize(node.getVideoSize())
                node.setVideoSize(size)
                videnc.setDefaultProfilePreset(node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            elif node.preview == nodeOutput:
                size = self.__calcEncodeableSize(node.getPreviewSize())
                node.setPreviewSize(size)
                videnc.setDefaultProfilePreset(node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            elif node.still == nodeOutput:
                size = self.__calcEncodeableSize(node.getStillSize())
                node.setStillSize(size)
                videnc.setDefaultProfilePreset(node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)

            nodeOutput.link(videnc.input)
        elif isinstance(node, dai.node.MonoCamera):
            videnc.setDefaultProfilePreset(node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            nodeOutput.link(videnc.input)
        elif isinstance(node, dai.node.StereoDepth):
            cameraNode = getattr(self.nodes, 'monoLeft', getattr(self.nodes, 'monoRight', None))
            if cameraNode is None:
                raise RuntimeError("Unable to find mono camera node to determine frame size!")
            videnc.setDefaultProfilePreset(cameraNode.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            nodeOutput.link(videnc.input)
        elif isinstance(node, dai.NeuralNetwork):
            w, h = self.__calcEncodeableSize(self.nnManager.inputSize)
            manip = self.pipeline.createImageManip()
            manip.initialConfig.setResize(w, h)

            videnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
            nodeOutput.link(manip.inputImage)
            manip.out.link(videnc.input)
        else:
            raise NotImplementedError("Unable to create mjpeg link for encountered node type: {}".format(type(node)))
        videnc.setQuality(self.poeQuality)
        videnc.bitstream.link(xout.input)

    def createColorCam(self,
        previewSize=None,
        res=dai.ColorCameraProperties.SensorResolution.THE_1080_P,
        fps=30,
        fullFov=True,
        ispScale: Optional[Tuple[int,int]] = None,
        orientation: dai.CameraImageOrientation=None,
        colorOrder=dai.ColorCameraProperties.ColorOrder.BGR,
        xout=False,
        xoutVideo=False,
        xoutStill=False,
        control=True, # Create input control
        pipeline=None,
        args=None,
        ) -> dai.node.ColorCamera:
        """
        Creates :obj:`depthai.node.ColorCamera` node based on specified attributes

        Args:
            previewSize (tuple, Optional): Size of the preview - :code:`(width, height)`
            res (depthai.ColorCameraProperties.SensorResolution, Optional): Camera resolution to be used
            fps (int, Optional): Camera FPS set on the device. Can limit / increase the amount of frames produced by the camera
            fullFov (bool, Optional): If set to :code:`True`, full frame will be scaled down to nn size. If to :code:`False`,
                it will first center crop the frame to meet the NN aspect ratio and then scale down the image.
            orientation (depthai.CameraImageOrientation, Optional): Custom camera orientation to be set on the device
            colorOrder (depthai.ColorCameraProperties, Optional): Color order to be used
            xout (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for this node
            xoutVideo (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for `video` output of this node
            xoutStill (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for `still` output of this node
            args (Object, Optional): Arguments from the ArgsManager
        """
        if pipeline is None:
            pipeline = self.pipeline

        if args is not None:
            self.nodes.camRgb = self.createColorCam(
                previewSize=(576, 320), # 1080P / 3
                res=args.rgbResolution,
                fps=args.rgbFps,
                orientation=dict(args.cameraOrientation).get(Previews.color.name),
                fullFov=not args.disableFullFovNn,
                xout=Previews.color.name in args.show,
                pipeline=pipeline,
                ispScale=args.ispScale
            )
            # Using CameraComponent's static function. Managers (including this one) will get deprecated when the full SDK
            # refactor is complete.
            from ..components.parser import parse_color_cam_control
            parse_color_cam_control(vars(args), self.nodes.camRgb)
            return self.nodes.camRgb

        self.nodes.camRgb = pipeline.createColorCamera()

        if previewSize is not None:
            self.nodes.camRgb.setPreviewSize(*previewSize)
        self.nodes.camRgb.setInterleaved(False)
        self.nodes.camRgb.setResolution(res)
        self.nodes.camRgb.setColorOrder(colorOrder)
        self.nodes.camRgb.setFps(fps)
        if orientation is not None:
            self.nodes.camRgb.setImageOrientation(orientation)
        self.nodes.camRgb.setPreviewKeepAspectRatio(not fullFov)

        if ispScale and len(ispScale) == 2:
            self.nodes.camRgb.setIspScale(int(ispScale[0]), int(ispScale[1]))

        if xout:
            self.nodes.xoutRgb = pipeline.createXLinkOut()
            self.nodes.xoutRgb.setStreamName(Previews.color.name)
            if self.lowBandwidth and not self.lowCapabilities:
                self._mjpegLink(self.nodes.camRgb, self.nodes.xoutRgb, self.nodes.camRgb.video)
            else:
                self.nodes.camRgb.preview.link(self.nodes.xoutRgb.input)
        if xoutVideo:
            self.nodes.xoutRgbVideo = pipeline.createXLinkOut()
            self.nodes.xoutRgbVideo.setStreamName(Previews.color.name + "_video")
            self.nodes.camRgb.video.link(self.nodes.xoutRgbVideo.input)
        if xoutStill:
            self.nodes.xoutRgbStill = pipeline.createXLinkOut()
            self.nodes.xoutRgbStill.setStreamName(Previews.color.name + "_still")
            self.nodes.camRgb.still.link(self.nodes.xoutRgbStill.input)

        if control:
            self.nodes.xinRgbControl = pipeline.createXLinkIn()
            self.nodes.xinRgbControl.setMaxDataSize(1024)
            self.nodes.xinRgbControl.setStreamName(Previews.color.name + "_control")
            self.nodes.xinRgbControl.out.link(self.nodes.camRgb.inputControl)

        return self.nodes.camRgb

    def createLeftCam(self,
        res=None,
        fps=30,
        orientation: dai.CameraImageOrientation=None,
        xout=False,
        control=True,
        pipeline = None,
        args=None,
        ) -> dai.node.MonoCamera:
        """
        Creates :obj:`depthai.node.MonoCamera` node based on specified attributes, assigned to :obj:`depthai.CameraBoardSocket.LEFT`

        Args:
            res (depthai.MonoCameraProperties.SensorResolution, Optional): Camera resolution to be used
            fps (int, Optional): Camera FPS set on the device. Can limit / increase the amount of frames produced by the camera
            orientation (depthai.CameraImageOrientation, Optional): Custom camera orientation to be set on the device
            xout (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for this node
            args (Object, Optional): Arguments from the ArgsManager
        """
        if pipeline is None:
            pipeline = self.pipeline

        if args is not None:
            return self.createLeftCam(
                args.monoResolution,
                args.monoFps,
                dict(args.cameraOrientation).get(Previews.left.name),
                Previews.left.name in args.show,
                pipeline=pipeline
            )

        self.nodes.monoLeft = pipeline.createMonoCamera()
        self.nodes.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        if orientation is not None:
            self.nodes.monoLeft.setImageOrientation(orientation)
        if res is None:
            self.nodes.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P if self.lowBandwidth and self.lowCapabilities else dai.MonoCameraProperties.SensorResolution.THE_720_P)
        else:
            self.nodes.monoLeft.setResolution(res)
        self.nodes.monoLeft.setFps(fps)

        if xout:
            self.nodes.xoutLeft = pipeline.createXLinkOut()
            self.nodes.xoutLeft.setStreamName(Previews.left.name)
            if self.lowBandwidth and not self.lowCapabilities:
                self._mjpegLink(self.nodes.monoLeft, self.nodes.xoutLeft, self.nodes.monoLeft.out)
            else:
                self.nodes.monoLeft.out.link(self.nodes.xoutLeft.input)
        if control:
            self.nodes.xinLeftControl = pipeline.createXLinkIn()
            self.nodes.xinLeftControl.setMaxDataSize(1024)
            self.nodes.xinLeftControl.setStreamName(Previews.left.name + "_control")
            self.nodes.xinLeftControl.out.link(self.nodes.monoLeft.inputControl)

        return self.nodes.monoLeft

    def createRightCam(self,
        res=None,
        fps=30,
        orientation: dai.CameraImageOrientation=None,
        xout=False,
        control=True,
        pipeline = None,
        args=None,
        ) -> dai.node.MonoCamera:
        """
        Creates :obj:`depthai.node.MonoCamera` node based on specified attributes, assigned to :obj:`depthai.CameraBoardSocket.RIGHT`

        Args:
            res (depthai.MonoCameraProperties.SensorResolution, Optional): Camera resolution to be used
            fps (int, Optional): Camera FPS set on the device. Can limit / increase the amount of frames produced by the camera
            orientation (depthai.CameraImageOrientation, Optional): Custom camera orientation to be set on the device
            xout (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for this node
            args (Object, Optional): Arguments from the ArgsManager
        """
        if pipeline is None:
            pipeline = self.pipeline

        if args is not None:
            return self.createRightCam(
                args.monoResolution,
                args.monoFps,
                dict(args.cameraOrientation).get(Previews.right.name),
                Previews.right.name in args.show,
                pipeline=pipeline
            )

        self.nodes.monoRight = pipeline.createMonoCamera()
        self.nodes.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        if orientation is not None:
            self.nodes.monoRight.setImageOrientation(orientation)
        if res is None:
            self.nodes.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P if self.lowBandwidth and self.lowCapabilities else dai.MonoCameraProperties.SensorResolution.THE_720_P)
        else:
            self.nodes.monoRight.setResolution(res)
        self.nodes.monoRight.setFps(fps)

        if xout:
            self.nodes.xoutRight = pipeline.createXLinkOut()
            self.nodes.xoutRight.setStreamName(Previews.right.name)
            if self.lowBandwidth and not self.lowCapabilities:
                self._mjpegLink(self.nodes.monoRight, self.nodes.xoutRight, self.nodes.monoRight.out)
            else:
                self.nodes.monoRight.out.link(self.nodes.xoutRight.input)

        if control:
            self.nodes.xinRightControl = pipeline.createXLinkIn()
            self.nodes.xinRightControl.setMaxDataSize(1024)
            self.nodes.xinRightControl.setStreamName(Previews.right.name + "_control")
            self.nodes.xinRightControl.out.link(self.nodes.monoRight.inputControl)

        return self.nodes.monoRight

    def updateIrConfig(self, device, irLaser=None, irFlood=None):
        """
        Updates IR configuration

        Args:
            irLaser (int, Optional): Sets the IR laser dot projector brightness (0..1200)
            irFlood (int, Optional): Sets the IR flood illuminator light brightness (0..1500)
        """
        if irLaser is not None:
            device.setIrLaserDotProjectorBrightness(irLaser)
        if irFlood is not None:
            device.setIrFloodLightBrightness(irFlood)

    def createDepth(self,
        dct=245,
        median=None,
        sigma=0,
        lr=True,
        lrcThreshold=5,
        extended=False,
        subpixel=False,
        useDisparity=False,
        useDepth=False,
        useRectifiedLeft=False,
        useRectifiedRight=False,
        runtimeSwitch=False,
        alignment=None,
        control=True,
        pipeline=None,
        args=None) -> dai.node.StereoDepth:
        """
        Creates :obj:`depthai.node.StereoDepth` node based on specified attributes

        Args:
            dct (int, Optional): Disparity Confidence Threshold (0..255). The less confident the network is, the more empty values
                are present in the depth map.
            median (depthai.MedianFilter, Optional): Median filter to be applied on the depth, use with :obj:`depthai.MedianFilter.MEDIAN_OFF` to disable median filtering
            sigma (int, Optional): Sigma value for bilateral filter (0..65535). If set to :code:`0`, the filter will be disabled.
            lr (bool, Optional): Set to :code:`True` to enable Left-Right Check
            lrcThreshold (int, Optional): Sets the Left-Right Check threshold value (0..10)
            extended (bool, Optional): Set to :code:`True` to enable the extended disparity
            subpixel (bool, Optional): Set to :code:`True` to enable the subpixel disparity
            useDisparity (bool, Optional): Set to :code:`True` to create output queue for disparity frames
            useDepth (bool, Optional): Set to :code:`True` to create output queue for depth frames
            useRectifiedLeft (bool, Optional): Set to :code:`True` to create output queue for rectified left frames
            useRectifiedRigh (bool, Optional): Set to :code:`True` to create output queue for rectified right frames
            runtimeSwitch (bool, Optional): Allows to change the depth configuration during the runtime but allocates resources for worst-case scenario (disabled by default)
            alignment (depthai.CameraBoardSocket, Optional): Aligns the depth map to the specified camera socket
            args (Object, Optional): Arguments from the ArgsManager

        Raises:
            RuntimeError: if left of right mono cameras were not initialized
        """
        if pipeline is None:
            pipeline = self.pipeline

        if args is not None:
            return self.createDepth(
                args.disparityConfidenceThreshold,
                self._getMedianFilter(args.stereoMedianSize),
                args.sigma,
                args.stereoLrCheck,
                args.lrcThreshold,
                args.extendedDisparity,
                args.subpixel,
                useDepth=Previews.depth.name in args.show or Previews.depthRaw.name in args.show,
                useDisparity=Previews.disparity.name in args.show or Previews.disparityColor.name in args.show,
                useRectifiedLeft=Previews.rectifiedLeft.name in args.show,
                useRectifiedRight=Previews.rectifiedRight.name in args.show,
                alignment=dai.CameraBoardSocket.RGB if args.stereoLrCheck and not args.noRgbDepthAlign else None,
                pipeline=pipeline
            )

        self.nodes.stereo = pipeline.createStereoDepth()

        self.nodes.stereo.initialConfig.setConfidenceThreshold(dct)
        if median is not None:
            self.nodes.stereo.initialConfig.setMedianFilter(median)
        self.nodes.stereo.initialConfig.setBilateralFilterSigma(sigma)
        self.nodes.stereo.initialConfig.setLeftRightCheckThreshold(lrcThreshold)

        self.nodes.stereo.setRuntimeModeSwitch(runtimeSwitch)
        self.nodes.stereo.setLeftRightCheck(lr)
        self.nodes.stereo.setExtendedDisparity(extended)
        self.nodes.stereo.setSubpixel(subpixel)
        if alignment is not None:
            self.nodes.stereo.setDepthAlign(alignment)
            if alignment == dai.CameraBoardSocket.RGB and 'camRgb' in dir(self.nodes):
                self.nodes.stereo.setOutputSize(*self.nodes.camRgb.getPreviewSize())
            elif alignment == dai.CameraBoardSocket.LEFT and 'monoLeft' in dir(self.nodes):
                self.nodes.stereo.setOutputSize(*self.nodes.monoLeft.getResolutionSize())
            elif alignment == dai.CameraBoardSocket.RIGHT and 'monoRight' in dir(self.nodes):
                self.nodes.stereo.setOutputSize(*self.nodes.monoRight.getResolutionSize())

        self._depthConfig = self.nodes.stereo.initialConfig.get()

        # Create mono left/right cameras if we haven't already
        if not hasattr(self.nodes, 'monoLeft'):
            raise RuntimeError("Left mono camera not initialized. Call createLeftCam(res, fps) first!")
        if not hasattr(self.nodes, 'monoRight'):
            raise RuntimeError("Right mono camera not initialized. Call createRightCam(res, fps) first!")

        self.nodes.monoLeft.out.link(self.nodes.stereo.left)
        self.nodes.monoRight.out.link(self.nodes.stereo.right)

        if control:
            self.nodes.xinStereoConfig = pipeline.createXLinkIn()
            self.nodes.xinStereoConfig.setMaxDataSize(1024)
            self.nodes.xinStereoConfig.setStreamName("stereoConfig")
            self.nodes.xinStereoConfig.out.link(self.nodes.stereo.inputConfig)

        if useDepth:
            self.nodes.xoutDepth = pipeline.createXLinkOut()
            self.nodes.xoutDepth.setStreamName(Previews.depthRaw.name)
            # if self.lowBandwidth and not self.lowCapabilities:  TODO change once depth frame type (14) is supported by VideoEncoder
            if False:
                self._mjpegLink(self.nodes.stereo, self.nodes.xoutDepth, self.nodes.stereo.depth)
            else:
                self.nodes.stereo.depth.link(self.nodes.xoutDepth.input)

        if useDisparity:
            self.nodes.xoutDisparity = pipeline.createXLinkOut()
            self.nodes.xoutDisparity.setStreamName(Previews.disparity.name)
            if False:
                self._mjpegLink(self.nodes.stereo, self.nodes.xoutDisparity, self.nodes.stereo.disparity)
            else:
                self.nodes.stereo.disparity.link(self.nodes.xoutDisparity.input)

        if useRectifiedLeft:
            self.nodes.xoutRectLeft = pipeline.createXLinkOut()
            self.nodes.xoutRectLeft.setStreamName(Previews.rectifiedLeft.name)
            # if self.lowBandwidth:  # disabled to limit the memory usage
            if False:
                self._mjpegLink(self.nodes.stereo, self.nodes.xoutRectLeft, self.nodes.stereo.rectifiedLeft)
            else:
                self.nodes.stereo.rectifiedLeft.link(self.nodes.xoutRectLeft.input)

        if useRectifiedRight:
            self.nodes.xoutRectRight = pipeline.createXLinkOut()
            self.nodes.xoutRectRight.setStreamName(Previews.rectifiedRight.name)
            # if self.lowBandwidth:  # disabled to limit the memory usage
            if False:
                self._mjpegLink(self.nodes.stereo, self.nodes.xoutRectRight, self.nodes.stereo.rectifiedRight)
            else:
                self.nodes.stereo.rectifiedRight.link(self.nodes.xoutRectRight.input)

        return self.nodes.stereo

    def _updateCamConfig(self, configRef: dai.CameraControl, exposure=None, sensitivity=None, saturation=None, contrast=None, brightness=None, sharpness=None):
        if any([exposure, sensitivity]):
            if not all([exposure, sensitivity]):
                raise RuntimeError("Both \"exposure\" and \"sensitivity\" arguments must be provided")
            configRef.setManualExposure(exposure, sensitivity)
        if saturation is not None:
            configRef.setSaturation(saturation)
        if sharpness is not None:
            configRef.setSharpness(sharpness)
        if contrast is not None:
            configRef.setContrast(contrast)
        if brightness is not None:
            configRef.setBrightness(brightness)

    def captureStill(self):
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        self._rgbConfigInputQueue.send(ctrl)

    # Added this function to send manual focus
    # configuration to inputControl queue.
    def setManualFocus(self, focus):
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(focus)
        self._rgbConfigInputQueue.send(ctrl)

    def triggerAutoFocus(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        self._rgbConfigInputQueue.send(ctrl)

    def triggerAutoExposure(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self._rgbConfigInputQueue.send(ctrl)

    def triggerAutoWhiteBalance(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        self._rgbConfigInputQueue.send(ctrl)

    def updateColorCamConfig(self, exposure=None, sensitivity=None, saturation=None, contrast=None, brightness=None, sharpness=None, autofocus=None, autowhitebalance=None, focus=None, whitebalance=None):
        """
        Updates :obj:`depthai.node.ColorCamera` node config

        Args:
            exposure (int, Optional): Exposure time in microseconds. Has to be set together with :obj:`sensitivity` (Usual range: 1..33000)
            sensitivity (int, Optional): Sensivity as ISO value. Has to be set together with :obj:`exposure` (Usual range: 100..1600)
            saturation (int, Optional): Image saturation (Allowed range: -10..10)
            contrast (int, Optional): Image contrast (Allowed range: -10..10)
            brightness (int, Optional): Image brightness (Allowed range: -10..10)
            sharpness (int, Optional): Image sharpness (Allowed range: 0..4)
            autofocus (dai.CameraControl.AutoFocusMode, Optional): Set the autofocus mode
            autowhitebalance (dai.CameraControl.AutoFocusMode, Optional): Set the autowhitebalance mode
            focus (int, Optional): Set the manual focus (lens position)
            whitebalance (int, Optional): Set the manual white balance
        """
        self._updateCamConfig(self._rgbConfig, exposure, sensitivity, saturation, contrast, brightness, sharpness)
        if autofocus is not None:
            self._rgbConfig.setAutoFocusMode(autofocus)
        if autowhitebalance is not None:
            self._rgbConfig.setAutoWhiteBalanceMode(autowhitebalance)
        if focus is not None:
            self._rgbConfig.setManualFocus(focus)
        if whitebalance is not None:
            self._rgbConfig.setManualWhiteBalance(whitebalance)

        if any([exposure, sensitivity, saturation, contrast, brightness, sharpness, autofocus, autowhitebalance, focus, whitebalance]):
            self._rgbConfigInputQueue.send(self._rgbConfig)

    def updateLeftCamConfig(self, exposure=None, sensitivity=None, saturation=None, contrast=None, brightness=None, sharpness=None):
        """
        Updates left :obj:`depthai.node.MonoCamera` node config

        Args:
            exposure (int, Optional): Exposure time in microseconds. Has to be set together with :obj:`sensitivity` (Usual range: 1..33000)
            sensitivity (int, Optional): Sensivity as ISO value. Has to be set together with :obj:`exposure` (Usual range: 100..1600)
            saturation (int, Optional): Image saturation (Allowed range: -10..10)
            contrast (int, Optional): Image contrast (Allowed range: -10..10)
            brightness (int, Optional): Image brightness (Allowed range: -10..10)
            sharpness (int, Optional): Image sharpness (Allowed range: 0..4)
        """
        self._updateCamConfig(self._leftConfig, exposure, sensitivity, saturation, contrast, brightness, sharpness)
        self._leftConfigInputQueue.send(self._leftConfig)

    def updateRightCamConfig(self, exposure=None, sensitivity=None, saturation=None, contrast=None, brightness=None, sharpness=None):
        """
        Updates right :obj:`depthai.node.MonoCamera` node config

        Args:
            exposure (int, Optional): Exposure time in microseconds. Has to be set together with :obj:`sensitivity` (Usual range: 1..33000)
            sensitivity (int, Optional): Sensivity as ISO value. Has to be set together with :obj:`exposure` (Usual range: 100..1600)
            saturation (int, Optional): Image saturation (Allowed range: -10..10)
            contrast (int, Optional): Image contrast (Allowed range: -10..10)
            brightness (int, Optional): Image brightness (Allowed range: -10..10)
            sharpness (int, Optional): Image sharpness (Allowed range: 0..4)
        """
        self._updateCamConfig(self._rightConfig, exposure, sensitivity, saturation, contrast, brightness, sharpness)
        self._rightConfigInputQueue.send(self._rightConfig)

    def updateDepthConfig(self, dct=None, sigma=None, median=None, lrcThreshold=None):
        """
        Updates :obj:`depthai.node.StereoDepth` node config

        Args:
            dct (int, Optional): Disparity Confidence Threshold (0..255). The less confident the network is, the more empty values
                are present in the depth map.
            median (depthai.MedianFilter, Optional): Median filter to be applied on the depth, use with :obj:`depthai.MedianFilter.MEDIANOFF` to disable median filtering
            sigma (int, Optional): Sigma value for bilateral filter (0..65535). If set to :code:`0`, the filter will be disabled.
            lrc (bool, Optional): Enables or disables Left-Right Check mode
            lrcThreshold (int, Optional): Sets the Left-Right Check threshold value (0..10)
        """
        if any([dct, sigma, median, lrcThreshold]):
            if dct is not None:
                self._depthConfig.costMatching.confidenceThreshold = dct
            if sigma is not None:
                self._depthConfig.postProcessing.bilateralSigmaValue = sigma
            if median is not None:
                self._depthConfig.postProcessing.median = median
            if lrcThreshold is not None:
                self._depthConfig.algorithmControl.leftRightCheckThreshold = lrcThreshold
            self._depthConfigInputQueue.send(self._depthConfig)

    def addNn(self, nn, xoutNnInput=False, xoutSbb=False):
        """
        Adds NN node to current pipeline. Usually obtained by calling :obj:`depthai_sdk.managers.NNetManager.createNN` method
        first

        Args:
            nn (depthai.node.NeuralNetwork): prepared NeuralNetwork node to be attached to the pipeline
            xoutNnInput (bool): Set to :code:`True` to create output queue for NN's passthough frames
            xoutSbb (bool): Set to :code:`True` to create output queue for Spatial Bounding Boxes (area that is used to calculate spatial location)
        """
        # TODO adjust this function once passthrough frame type (8) is supported by VideoEncoder (for self.MjpegLink)
        if xoutNnInput:
            self.nodes.xoutNnInput = self.pipeline.createXLinkOut()
            self.nodes.xoutNnInput.setStreamName(Previews.nnInput.name)
            nn.passthrough.link(self.nodes.xoutNnInput.input)

        if xoutSbb and self.nnManager._nnFamily in ("YOLO", "mobilenet"):
            self.nodes.xoutSbb = self.pipeline.createXLinkOut()
            self.nodes.xoutSbb.setStreamName("sbb")
            nn.boundingBoxMapping.link(self.nodes.xoutSbb.input)

    def createSystemLogger(self, rate=1):
        """
        Creates :obj:`depthai.node.SystemLogger` node together with XLinkOut

        Args:
            rate (int, Optional): Specify logging rate (in Hz)
        """
        self.nodes.systemLogger = self.pipeline.createSystemLogger()
        self.nodes.systemLogger.setRate(rate)
        self.nodes.xoutSystemLogger = self.pipeline.createXLinkOut()
        self.nodes.xoutSystemLogger.setStreamName("systemLogger")
        self.nodes.systemLogger.out.link(self.nodes.xoutSystemLogger.input)

    def createEncoder(self, cameraName, encFps=30, encQuality=100):
        """
        Creates H.264 / H.265 video encoder (:obj:`depthai.node.VideoEncoder` instance)

        Args:
            cameraName (str): Camera name to create the encoder for
            encFps (int, Optional): Specify encoding FPS
            encQuality (int, Optional): Specify encoding quality (1-100)

        Raises:
            ValueError: if cameraName is not a supported camera name
            RuntimeError: if specified camera node was not present
        """
        allowedSources = [Previews.left.name, Previews.right.name, Previews.color.name]
        if cameraName not in allowedSources:
            raise ValueError(
                "Camera param invalid, received {}, available choices: {}".format(cameraName, allowedSources))
        nodeName = cameraName.lower() + 'Enc'
        xoutName = nodeName + "Xout"
        encProfile = dai.VideoEncoderProperties.Profile.H264_MAIN

        if cameraName == Previews.color.name:
            if not hasattr(self.nodes, 'camRgb'):
                raise RuntimeError("RGB camera not initialized. Call createColorCam(res, fps) first!")
            encProfile = dai.VideoEncoderProperties.Profile.H264_MAIN
            encIn = self.nodes.camRgb.video

        elif cameraName == Previews.left.name:
            if not hasattr(self.nodes, 'monoLeft'):
                raise RuntimeError("Left mono camera not initialized. Call createLeftCam(res, fps) first!")
            encIn = self.nodes.monoLeft.out
        elif cameraName == Previews.right.name:
            if not hasattr(self.nodes, 'monoRight'):
                raise RuntimeError("Right mono camera not initialized. Call createRightCam(res, fps) first!")
            encIn = self.nodes.monoRight.out

        enc = self.pipeline.createVideoEncoder()
        enc.setDefaultProfilePreset(encFps, encProfile)
        enc.setQuality(encQuality)
        encIn.link(enc.input)
        setattr(self.nodes, nodeName, enc)

        encXout = self.pipeline.createXLinkOut()
        enc.bitstream.link(encXout.input)
        encXout.setStreamName(xoutName)
        setattr(self.nodes, xoutName, encXout)

    def enableLowBandwidth(self, poeQuality):
        """
        Enables low-bandwidth mode

        Args:
            poeQuality (int, Optional): PoE encoding quality, can decrease frame quality but decrease latency
        """
        self.lowBandwidth = True
        self.poeQuality = poeQuality

    def setXlinkChunkSize(self, chunkSize):
        self.pipeline.setXLinkChunkSize(chunkSize)

    def setCameraTuningBlob(self, path):
        self.pipeline.setCameraTuningBlobPath(path)

    def _getMedianFilter(self, size: int) -> dai.MedianFilter:
        if size == 3:
            return dai.MedianFilter.KERNEL_3x3
        elif size == 5:
            return dai.MedianFilter.KERNEL_5x5
        elif size == 7:
            return dai.MedianFilter.KERNEL_7x7
        else:
            return dai.MedianFilter.MEDIAN_OFF


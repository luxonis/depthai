from types import SimpleNamespace
import depthai as dai

from ..previews import Previews


class PipelineManager:
    """
    Manager class handling different :obj:`depthai.Pipeline` operations. Most of the functions wrap up nodes creation
    and connection logic onto a set of convenience functions.
    """

    def __init__(self, openvino_version=None):
        self.openvino_version=openvino_version

        if openvino_version is not None:
            self.pipeline.setOpenVINOVersion(openvino_version)

    #: depthai.OpenVINO.Version: OpenVINO version which will be used in pipeline
    openvino_version = None
    #: bool: If set to :code:`True`, manager will MJPEG-encode the packets sent from device to host to lower the bandwidth usage. **Can break** if more than 3 encoded outputs requested
    lowBandwidth = False
    #: depthai.Pipeline: Ready to use requested pipeline. Can be passed to :obj:`depthai.Device` to start execution
    pipeline = dai.Pipeline()
    #: types.SimpleNamespace: Contains all nodes added to the :attr:`pipeline` object, can be used to conveniently access nodes by their name
    nodes = SimpleNamespace()

    _depthConfig = dai.StereoDepthConfig()

    def set_nn_manager(self, nn_manager):
        """
        Assigns NN manager. It also syncs the pipeline versions between those two objects

        Args:
            nn_manager (depthai_sdk.managers.NNetManager): NN manager instance
        """
        self.nn_manager = nn_manager
        if self.openvino_version is None and self.nn_manager.openvino_version:
            self.pipeline.setOpenVINOVersion(self.nn_manager.openvino_version)
        else:
            self.nn_manager.openvino_version = self.pipeline.getOpenVINOVersion()

    def create_default_queues(self, device):
        """
        Creates queues for all requested XLinkOut's and XLinkIn's.

        Args:
            device (depthai.Device): Running device instance
        """
        for xout in filter(lambda node: isinstance(node, dai.node.XLinkOut), vars(self.nodes).values()):
            device.getOutputQueue(xout.getStreamName(), maxSize=1, blocking=False)
        for xin in filter(lambda node: isinstance(node, dai.node.XLinkIn), vars(self.nodes).values()):
            device.getInputQueue(xin.getStreamName(), maxSize=1, blocking=False)

    def __calc_encodeable_size(self, source_size):
        w, h = source_size
        if w % 16 > 0:
            new_w = w - (w % 16)
            h = int((new_w / w) * h)
            w = int(new_w)
        if h % 2 > 0:
            h -= 1
        return w, h

    def _mjpeg_link(self, node, xout, node_output):
        print("Creating MJPEG link for {} node and {} xlink stream...".format(node.getName(), xout.getStreamName()))
        videnc = self.pipeline.createVideoEncoder()
        if isinstance(node, dai.node.ColorCamera):
            if node.video == node_output:
                size = self.__calc_encodeable_size(node.getVideoSize())
                node.setVideoSize(size)
                videnc.setDefaultProfilePreset(*size, node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            elif node.preview == node_output:
                size = self.__calc_encodeable_size(node.getPreviewSize())
                node.setPreviewSize(size)
                videnc.setDefaultProfilePreset(*size, node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            elif node.still == node_output:
                size = self.__calc_encodeable_size(node.getStillSize())
                node.setStillSize(size)
                videnc.setDefaultProfilePreset(*size, node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)

            node_output.link(videnc.input)
        elif isinstance(node, dai.node.MonoCamera):
            videnc.setDefaultProfilePreset(node.getResolutionWidth(), node.getResolutionHeight(), node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            node_output.link(videnc.input)
        elif isinstance(node, dai.node.StereoDepth):
            camera_node = getattr(self.nodes, 'mono_left', getattr(self.nodes, 'mono_right', None))
            if camera_node is None:
                raise RuntimeError("Unable to find mono camera node to determine frame size!")
            videnc.setDefaultProfilePreset(camera_node.getResolutionWidth(), camera_node.getResolutionHeight(), camera_node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            node_output.link(videnc.input)
        elif isinstance(node, dai.NeuralNetwork):
            w, h = self.__calc_encodeable_size(self.nn_manager.input_size)
            manip = self.pipeline.createImageManip()
            manip.initialConfig.setResize(w, h)

            videnc.setDefaultProfilePreset(w, h, 30, dai.VideoEncoderProperties.Profile.MJPEG)
            node_output.link(manip.inputImage)
            manip.out.link(videnc.input)
        else:
            raise NotImplementedError("Unable to create mjpeg link for encountered node type: {}".format(type(node)))
        videnc.bitstream.link(xout.input)

    def create_color_cam(self, preview_size=None, res=dai.ColorCameraProperties.SensorResolution.THE_1080_P, fps=30, full_fov=True, orientation: dai.CameraImageOrientation=None, xout=False):
        """
        Creates :obj:`depthai.node.ColorCamera` node based on specified attributes

        Args:
            preview_size (tuple, Optional): Size of the preview output - :code:`(width, height)`. Usually same as NN input
            res (depthai.ColorCameraProperties.SensorResolution, Optional): Camera resolution to be used
            fps (int, Optional): Camera FPS set on the device. Can limit / increase the amount of frames produced by the camera
            full_fov (bool, Optional): If set to :code:`True`, full frame will be scaled down to nn size. If to :code:`False`,
                it will first center crop the frame to meet the NN aspect ratio and then scale down the image.
            orientation (depthai.CameraImageOrientation, Optional): Custom camera orientation to be set on the device
            xout (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for this node
        """
        self.nodes.cam_rgb = self.pipeline.createColorCamera()
        if preview_size is not None:
            self.nodes.cam_rgb.setPreviewSize(*preview_size)
        self.nodes.cam_rgb.setInterleaved(False)
        self.nodes.cam_rgb.setResolution(res)
        self.nodes.cam_rgb.setFps(fps)
        if orientation is not None:
            self.nodes.cam_rgb.setImageOrientation(orientation)
        self.nodes.cam_rgb.setPreviewKeepAspectRatio(not full_fov)
        self.nodes.xout_rgb = self.pipeline.createXLinkOut()
        self.nodes.xout_rgb.setStreamName(Previews.color.name)
        if xout:
            if self.lowBandwidth:
                self._mjpeg_link(self.nodes.cam_rgb, self.nodes.xout_rgb, self.nodes.cam_rgb.video)
            else:
                self.nodes.cam_rgb.video.link(self.nodes.xout_rgb.input)


    def create_left_cam(self, res=dai.MonoCameraProperties.SensorResolution.THE_720_P, fps=30, orientation: dai.CameraImageOrientation=None, xout=False):
        """
        Creates :obj:`depthai.node.MonoCamera` node based on specified attributes, assigned to :obj:`depthai.CameraBoardSocket.LEFT`

        Args:
            res (depthai.MonoCameraProperties.SensorResolution, Optional): Camera resolution to be used
            fps (int, Optional): Camera FPS set on the device. Can limit / increase the amount of frames produced by the camera
            orientation (depthai.CameraImageOrientation, Optional): Custom camera orientation to be set on the device
            xout (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for this node
        """
        self.nodes.mono_left = self.pipeline.createMonoCamera()
        self.nodes.mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        if orientation is not None:
            self.nodes.mono_left.setImageOrientation(orientation)
        self.nodes.mono_left.setResolution(res)
        self.nodes.mono_left.setFps(fps)

        self.nodes.xout_left = self.pipeline.createXLinkOut()
        self.nodes.xout_left.setStreamName(Previews.left.name)
        if xout:
            if self.lowBandwidth:
                self._mjpeg_link(self.nodes.mono_left, self.nodes.xout_left, self.nodes.mono_left.out)
            else:
                self.nodes.mono_left.out.link(self.nodes.xout_left.input)

    def create_right_cam(self, res=dai.MonoCameraProperties.SensorResolution.THE_720_P, fps=30, orientation: dai.CameraImageOrientation=None, xout=False):
        """
        Creates :obj:`depthai.node.MonoCamera` node based on specified attributes, assigned to :obj:`depthai.CameraBoardSocket.RIGHT`

        Args:
            res (depthai.MonoCameraProperties.SensorResolution, Optional): Camera resolution to be used
            fps (int, Optional): Camera FPS set on the device. Can limit / increase the amount of frames produced by the camera
            orientation (depthai.CameraImageOrientation, Optional): Custom camera orientation to be set on the device
            xout (bool, Optional): If set to :code:`True`, a dedicated :obj:`depthai.node.XLinkOut` will be created for this node
        """
        self.nodes.mono_right = self.pipeline.createMonoCamera()
        self.nodes.mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        if orientation is not None:
            self.nodes.mono_right.setImageOrientation(orientation)
        self.nodes.mono_right.setResolution(res)
        self.nodes.mono_right.setFps(fps)

        self.nodes.xout_right = self.pipeline.createXLinkOut()
        self.nodes.xout_right.setStreamName(Previews.right.name)
        if xout:
            if self.lowBandwidth:
                self._mjpeg_link(self.nodes.mono_right, self.nodes.xout_right, self.nodes.mono_right.out)
            else:
                self.nodes.mono_right.out.link(self.nodes.xout_right.input)

    def create_depth(self, dct=245, median=dai.MedianFilter.KERNEL_7x7, sigma=0, lr=False, lrc_threshold=4, extended=False, subpixel=False, useDisparity=False, useDepth=False, useRectifiedLeft=False, useRectifiedRight=False):
        """
        Creates :obj:`depthai.node.StereoDepth` node based on specified attributes

        Args:
            dct (int, Optional): Disparity Confidence Threshold (0..255). The less confident the network is, the more empty values
                are present in the depth map.
            median (depthai.MedianFilter, Optional): Median filter to be applied on the depth, use with :obj:`depthai.MedianFilter.MEDIAN_OFF` to disable median filtering
            sigma (int, Optional): Sigma value for bilateral filter (0..65535). If set to :code:`0`, the filter will be disabled.
            lr (bool, Optional): Set to :code:`True` to enable Left-Right Check
            lrc_threshold (int, Optional): Sets the Left-Right Check threshold value (0..10)
            extended (bool, Optional): Set to :code:`True` to enable the extended disparity
            subpixel (bool, Optional): Set to :code:`True` to enable the subpixel disparity
            useDisparity (bool, Optional): Set to :code:`True` to create output queue for disparity frames
            useDepth (bool, Optional): Set to :code:`True` to create output queue for depth frames
            useRectifiedLeft (bool, Optional): Set to :code:`True` to create output queue for rectified left frames
            useRectifiedRigh (bool, Optional): Set to :code:`True` to create output queue for rectified righ frames

        Raises:
            RuntimeError: if left of right mono cameras were not initialized
        """
        self.nodes.stereo = self.pipeline.createStereoDepth()

        self.nodes.stereo.initialConfig.setConfidenceThreshold(dct)
        self._depthConfig.setConfidenceThreshold(dct)
        self.nodes.stereo.initialConfig.setMedianFilter(median)
        self._depthConfig.setMedianFilter(median)
        self.nodes.stereo.initialConfig.setBilateralFilterSigma(sigma)
        self._depthConfig.setBilateralFilterSigma(sigma)
        self.nodes.stereo.initialConfig.setLeftRightCheckThreshold(lrc_threshold)
        self._depthConfig.setLeftRightCheckThreshold(lrc_threshold)

        self.nodes.stereo.setLeftRightCheck(lr)
        self.nodes.stereo.setExtendedDisparity(extended)
        self.nodes.stereo.setSubpixel(subpixel)

        # Create mono left/right cameras if we haven't already
        if not hasattr(self.nodes, 'mono_left'):
            raise RuntimeError("Left mono camera not initialized. Call create_left_cam(res, fps) first!")
        if not hasattr(self.nodes, 'mono_right'):
            raise RuntimeError("Right mono camera not initialized. Call create_right_cam(res, fps) first!")

        self.nodes.mono_left.out.link(self.nodes.stereo.left)
        self.nodes.mono_right.out.link(self.nodes.stereo.right)

        self.nodes.xin_stereo_config = self.pipeline.createXLinkIn()
        self.nodes.xin_stereo_config.setStreamName("stereo_config")
        self.nodes.xin_stereo_config.out.link(self.nodes.stereo.inputConfig)

        if useDepth:
            self.nodes.xout_depth = self.pipeline.createXLinkOut()
            self.nodes.xout_depth.setStreamName(Previews.depth_raw.name)
            # if self.lowBandwidth:  TODO change once depth frame type (14) is supported by VideoEncoder
            if False:
                self._mjpeg_link(self.nodes.stereo, self.nodes.xout_depth, self.nodes.stereo.depth)
            else:
                self.nodes.stereo.depth.link(self.nodes.xout_depth.input)

        if useDisparity:
            self.nodes.xout_disparity = self.pipeline.createXLinkOut()
            self.nodes.xout_disparity.setStreamName(Previews.disparity.name)
            if self.lowBandwidth:
                self._mjpeg_link(self.nodes.stereo, self.nodes.xout_disparity, self.nodes.stereo.disparity)
            else:
                self.nodes.stereo.disparity.link(self.nodes.xout_disparity.input)

        if useRectifiedLeft:
            self.nodes.xout_rect_left = self.pipeline.createXLinkOut()
            self.nodes.xout_rect_left.setStreamName(Previews.rectified_left.name)
            if self.lowBandwidth:
                self._mjpeg_link(self.nodes.stereo, self.nodes.xout_rect_left, self.nodes.stereo.rectifiedLeft)
            else:
                self.nodes.stereo.rectifiedLeft.link(self.nodes.xout_rect_left.input)

        if useRectifiedRight:
            self.nodes.xout_rect_right = self.pipeline.createXLinkOut()
            self.nodes.xout_rect_right.setStreamName(Previews.rectified_right.name)
            if self.lowBandwidth:
                self._mjpeg_link(self.nodes.stereo, self.nodes.xout_rect_right, self.nodes.stereo.rectifiedRight)
            else:
                self.nodes.stereo.rectifiedRight.link(self.nodes.xout_rect_right.input)

    def update_depth_config(self, device, dct=None, sigma=None, median=None, lrc_threshold=None):
        """
        Updates :obj:`depthai.node.StereoDepth` node config

        Args:
            device (depthai.Device): Running device instance
            dct (int, Optional): Disparity Confidence Threshold (0..255). The less confident the network is, the more empty values
                are present in the depth map.
            median (depthai.MedianFilter, Optional): Median filter to be applied on the depth, use with :obj:`depthai.MedianFilter.MEDIAN_OFF` to disable median filtering
            sigma (int, Optional): Sigma value for bilateral filter (0..65535). If set to :code:`0`, the filter will be disabled.
            lrc_threshold (int, Optional): Sets the Left-Right Check threshold value (0..10)
        """
        if dct is not None:
            self._depthConfig.setConfidenceThreshold(dct)
        if sigma is not None:
            self._depthConfig.setBilateralFilterSigma(sigma)
        if median is not None:
            self._depthConfig.setMedianFilter(median)
        if lrc_threshold is not None:
            self._depthConfig.setLeftRightCheckThreshold(lrc_threshold)

        device.getInputQueue("stereo_config").send(self._depthConfig)

    def add_nn(self, nn, sync=False, use_depth=False, xout_nn_input=False, xout_sbb=False):
        """
        Adds NN node to current pipeline. Usually obtained by calling :obj:`depthai_sdk.managers.NNetManager.create_nn_pipeline` method
        first

        Args:
            nn (depthai.node.NeuralNetwork): prepared NeuralNetwork node to be attached to the pipeline
            sync (bool): Will attach NN's passthough output to source XLinkOut, making the frame appear in the output queue same time as NN-results packet
            use_depth (bool): If used together with :code:`sync` flag, will attach NN's passthoughDepth output to depth XLinkOut, making the depth frame appear in the output queue same time as NN-results packet
            xout_nn_input (bool): Set to :code:`True` to create output queue for NN's passthough frames
            xout_sbb (bool): Set to :code:`True` to create output queue for Spatial Bounding Boxes (area that is used to calculate spatial location)
        """
        # TODO adjust this function once passthrough frame type (8) is supported by VideoEncoder (for self._mjpeg_link)
        if xout_nn_input or (sync and self.nn_manager.source == "host"):
            self.nodes.xout_nn_input = self.pipeline.createXLinkOut()
            self.nodes.xout_nn_input.setStreamName(Previews.nn_input.name)
            nn.passthrough.link(self.nodes.xout_nn_input.input)

        if xout_sbb and self.nn_manager._nn_family in ("YOLO", "mobilenet"):
            self.nodes.xout_sbb = self.pipeline.createXLinkOut()
            self.nodes.xout_sbb.setStreamName("sbb")
            nn.boundingBoxMapping.link(self.nodes.xout_sbb.input)

        if sync:
            if self.nn_manager.source == "color":
                if not hasattr(self.nodes, "xout_rgb"):
                    self.nodes.xout_rgb = self.pipeline.createXLinkOut()
                    self.nodes.xout_rgb.setStreamName(Previews.color.name)
                nn.passthrough.link(self.nodes.xout_rgb.input)
            elif self.nn_manager.source == "left":
                if not hasattr(self.nodes, "xout_left"):
                    self.nodes.xout_left = self.pipeline.createXLinkOut()
                    self.nodes.xout_left.setStreamName(Previews.left.name)
                nn.passthrough.link(self.nodes.xout_left.input)
            elif self.nn_manager.source == "right":
                if not hasattr(self.nodes, "xout_right"):
                    self.nodes.xout_right = self.pipeline.createXLinkOut()
                    self.nodes.xout_right.setStreamName(Previews.right.name)
                nn.passthrough.link(self.nodes.xout_right.input)
            elif self.nn_manager.source == "rectified_left":
                if not hasattr(self.nodes, "xout_rect_left"):
                    self.nodes.xout_rect_left = self.pipeline.createXLinkOut()
                    self.nodes.xout_rect_left.setStreamName(Previews.rectified_left.name)
                nn.passthrough.link(self.nodes.xout_rect_left.input)
            elif self.nn_manager.source == "rectified_right":
                if not hasattr(self.nodes, "xout_rect_right"):
                    self.nodes.xout_rect_right = self.pipeline.createXLinkOut()
                    self.nodes.xout_rect_right.setStreamName(Previews.rectified_right.name)
                nn.passthrough.link(self.nodes.xout_rect_right.input)

            if self.nn_manager._nn_family in ("YOLO", "mobilenet") and use_depth:
                if not hasattr(self.nodes, "xout_depth"):
                    self.nodes.xout_depth = self.pipeline.createXLinkOut()
                    self.nodes.xout_depth.setStreamName(Previews.depth.name)
                nn.passthroughDepth.link(self.nodes.xout_depth.input)

    def create_system_logger(self, rate=1):
        """
        Creates :obj:`depthai.node.SystemLogger` node together with XLinkOut

        Args:
            rate (int, Optional): Specify logging rate (in Hz)
        """
        self.nodes.system_logger = self.pipeline.createSystemLogger()
        self.nodes.system_logger.setRate(1)
        self.nodes.xout_system_logger = self.pipeline.createXLinkOut()
        self.nodes.xout_system_logger.setStreamName("system_logger")
        self.nodes.system_logger.out.link(self.nodes.xout_system_logger.input)

    def create_encoder(self, camera_name, enc_fps=30):
        """
        Creates H.264 / H.265 video encoder (:obj:`depthai.node.VideoEncoder` instance)

        Args:
            camera_name (str): Camera name to create the encoder for
            enc_fps (int, Optional): Specify encoding FPS

        Raises:
            ValueError: if camera_name is not a supported camera name
            RuntimeError: if specified camera node was not present
        """
        allowed_sources = [Previews.left.name, Previews.right.name, Previews.color.name]
        if camera_name not in allowed_sources:
            raise ValueError(
                "Camera param invalid, received {}, available choices: {}".format(camera_name, allowed_sources))
        node_name = camera_name.lower() + '_enc'
        xout_name = node_name + "_xout"
        enc_profile = dai.VideoEncoderProperties.Profile.H264_MAIN

        if camera_name == Previews.color.name:
            if not hasattr(self.nodes, 'cam_rgb'):
                raise RuntimeError("RGB camera not initialized. Call create_color_cam(res, fps) first!")
            enc_resolution = (self.nodes.cam_rgb.getVideoWidth(), self.nodes.cam_rgb.getVideoHeight())
            enc_profile = dai.VideoEncoderProperties.Profile.H265_MAIN
            enc_in = self.nodes.cam_rgb.video

        elif camera_name == Previews.left.name:
            if not hasattr(self.nodes, 'mono_left'):
                raise RuntimeError("Left mono camera not initialized. Call create_left_cam(res, fps) first!")
            enc_resolution = (
            self.nodes.mono_left.getResolutionWidth(), self.nodes.mono_left.getResolutionHeight())
            enc_in = self.nodes.mono_left.out
        elif camera_name == Previews.right.name:
            if not hasattr(self.nodes, 'mono_right'):
                raise RuntimeError("Right mono camera not initialized. Call create_right_cam(res, fps) first!")
            enc_resolution = (
            self.nodes.mono_right.getResolutionWidth(), self.nodes.mono_right.getResolutionHeight())
            enc_in = self.nodes.mono_right.out

        enc = self.pipeline.createVideoEncoder()
        enc.setDefaultProfilePreset(*enc_resolution, enc_fps, enc_profile)
        enc_in.link(enc.input)
        setattr(self.nodes, node_name, enc)

        enc_xout = self.pipeline.createXLinkOut()
        enc.bitstream.link(enc_xout.input)
        enc_xout.setStreamName(xout_name)
        setattr(self.nodes, xout_name, enc_xout)

    def enableLowBandwidth(self):
        """
        Enables low-bandwidth mode.
        """
        self.lowBandwidth = True

    def set_xlink_chunk_size(self, chunk_size):
        self.pipeline.setXLinkChunkSize(chunk_size)

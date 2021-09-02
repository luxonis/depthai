from types import SimpleNamespace
import depthai as dai

from ..previews import Previews


class PipelineManager:
    pipeline = dai.Pipeline()
    nodes = SimpleNamespace()
    depthConfig = dai.StereoDepthConfig()

    def __init__(self, openvino_version=None, lowBandwidth=False):
        self.openvino_version=openvino_version
        if openvino_version is not None:
            self.pipeline.setOpenVINOVersion(openvino_version)
        self.lowBandwidth = lowBandwidth

    def set_nn_manager(self, nn_manager):
        self.nn_manager = nn_manager
        if self.openvino_version is None and self.nn_manager.openvino_version:
            self.pipeline.setOpenVINOVersion(self.nn_manager.openvino_version)
        else:
            self.nn_manager.openvino_version = self.pipeline.getOpenVINOVersion()

    def create_default_queues(self, device):
        for xout in filter(lambda node: isinstance(node, dai.node.XLinkOut), vars(self.nodes).values()):
            device.getOutputQueue(xout.getStreamName(), maxSize=1, blocking=False)
        for xin in filter(lambda node: isinstance(node, dai.node.XLinkIn), vars(self.nodes).values()):
            device.getInputQueue(xin.getStreamName(), maxSize=1, blocking=False)

    def mjpeg_link(self, node, xout, node_output):
        print("Creating MJPEG link for {} node and {} xlink stream...".format(node.getName(), xout.getStreamName()))
        videnc = self.pipeline.createVideoEncoder()
        if isinstance(node, dai.node.ColorCamera) or isinstance(node, dai.node.MonoCamera):
            videnc.setDefaultProfilePreset(node.getResolutionWidth(), node.getResolutionHeight(), node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            node_output.link(videnc.input)
        elif isinstance(node, dai.node.StereoDepth):
            camera_node = getattr(self.nodes, 'mono_left', getattr(self.nodes, 'mono_right', None))
            if camera_node is None:
                raise RuntimeError("Unable to find mono camera node to determine frame size!")
            videnc.setDefaultProfilePreset(camera_node.getResolutionWidth(), camera_node.getResolutionHeight(), camera_node.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            node_output.link(videnc.input)
        elif isinstance(node, dai.NeuralNetwork):
            w, h = self.nn_manager.input_size
            if w % 16 > 0:
                new_w = w - (w % 16)
                h = int((new_w / w) * h)
                w = int(new_w)
            if h % 2 > 0:
                h -= 1
            manip = self.pipeline.createImageManip()
            manip.initialConfig.setResize(w, h)

            videnc.setDefaultProfilePreset(w, h, 30, dai.VideoEncoderProperties.Profile.MJPEG)
            node_output.link(manip.inputImage)
            manip.out.link(videnc.input)
        else:
            raise NotImplementedError("Unable to create mjpeg link for encountered node type: {}".format(type(node)))
        videnc.bitstream.link(xout.input)

    def create_color_cam(self, preview_size=None, res=dai.ColorCameraProperties.SensorResolution.THE_1080_P, fps=30, full_fov=True, xout=False):
        # Define a source - color camera
        self.nodes.cam_rgb = self.pipeline.createColorCamera()
        if preview_size is not None:
            self.nodes.cam_rgb.setPreviewSize(*preview_size)
        self.nodes.cam_rgb.setInterleaved(False)
        self.nodes.cam_rgb.setResolution(res)
        self.nodes.cam_rgb.setFps(fps)
        self.nodes.cam_rgb.setPreviewKeepAspectRatio(not full_fov)
        self.nodes.xout_rgb = self.pipeline.createXLinkOut()
        self.nodes.xout_rgb.setStreamName(Previews.color.name)
        if xout:
            if self.lowBandwidth:
                self.mjpeg_link(self.nodes.cam_rgb, self.nodes.xout_rgb, self.nodes.cam_rgb.video)
            else:
                self.nodes.cam_rgb.video.link(self.nodes.xout_rgb.input)

    def create_depth(self, dct=245, median=dai.MedianFilter.KERNEL_7x7, sigma=0, lr=False, lrc_threshold=4, extended=False, subpixel=False, useDisparity=False, useDepth=False, useRectifiedLeft=False, useRectifiedRight=False):
        self.nodes.stereo = self.pipeline.createStereoDepth()

        self.nodes.stereo.initialConfig.setConfidenceThreshold(dct)
        self.depthConfig.setConfidenceThreshold(dct)
        self.nodes.stereo.initialConfig.setMedianFilter(median)
        self.depthConfig.setMedianFilter(median)
        self.nodes.stereo.initialConfig.setBilateralFilterSigma(sigma)
        self.depthConfig.setBilateralFilterSigma(sigma)
        self.nodes.stereo.initialConfig.setLeftRightCheckThreshold(lrc_threshold)
        self.depthConfig.setLeftRightCheckThreshold(lrc_threshold)

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
                self.mjpeg_link(self.nodes.stereo, self.nodes.xout_depth, self.nodes.stereo.depth)
            else:
                self.nodes.stereo.depth.link(self.nodes.xout_depth.input)

        if useDisparity:
            self.nodes.xout_disparity = self.pipeline.createXLinkOut()
            self.nodes.xout_disparity.setStreamName(Previews.disparity.name)
            if self.lowBandwidth:
                self.mjpeg_link(self.nodes.stereo, self.nodes.xout_disparity, self.nodes.stereo.disparity)
            else:
                self.nodes.stereo.disparity.link(self.nodes.xout_disparity.input)

        if useRectifiedLeft:
            self.nodes.xout_rect_left = self.pipeline.createXLinkOut()
            self.nodes.xout_rect_left.setStreamName(Previews.rectified_left.name)
            if self.lowBandwidth:
                self.mjpeg_link(self.nodes.stereo, self.nodes.xout_rect_left, self.nodes.stereo.rectifiedLeft)
            else:
                self.nodes.stereo.rectifiedLeft.link(self.nodes.xout_rect_left.input)

        if useRectifiedRight:
            self.nodes.xout_rect_right = self.pipeline.createXLinkOut()
            self.nodes.xout_rect_right.setStreamName(Previews.rectified_right.name)
            if self.lowBandwidth:
                self.mjpeg_link(self.nodes.stereo, self.nodes.xout_rect_right, self.nodes.stereo.rectifiedRight)
            else:
                self.nodes.stereo.rectifiedRight.link(self.nodes.xout_rect_right.input)

    def update_depth_config(self, device, dct=None, sigma=None, median=None, lrc_threshold=None):
        if dct is not None:
            self.depthConfig.setConfidenceThreshold(dct)
        if sigma is not None:
            self.depthConfig.setBilateralFilterSigma(sigma)
        if median is not None:
            self.depthConfig.setMedianFilter(median)
        if lrc_threshold is not None:
            self.depthConfig.setLeftRightCheckThreshold(lrc_threshold)

        device.getInputQueue("stereo_config").send(self.depthConfig)


    def create_left_cam(self, res=dai.MonoCameraProperties.SensorResolution.THE_720_P, fps=30, xout=False):
        self.nodes.mono_left = self.pipeline.createMonoCamera()
        self.nodes.mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.nodes.mono_left.setResolution(res)
        self.nodes.mono_left.setFps(fps)

        self.nodes.xout_left = self.pipeline.createXLinkOut()
        self.nodes.xout_left.setStreamName(Previews.left.name)
        if xout:
            if self.lowBandwidth:
                self.mjpeg_link(self.nodes.mono_left, self.nodes.xout_left, self.nodes.mono_left.out)
            else:
                self.nodes.mono_left.out.link(self.nodes.xout_left.input)

    def create_right_cam(self, res=dai.MonoCameraProperties.SensorResolution.THE_720_P, fps=30, xout=False):
        self.nodes.mono_right = self.pipeline.createMonoCamera()
        self.nodes.mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.nodes.mono_right.setResolution(res)
        self.nodes.mono_right.setFps(fps)

        self.nodes.xout_right = self.pipeline.createXLinkOut()
        self.nodes.xout_right.setStreamName(Previews.right.name)
        if xout:
            if self.lowBandwidth:
                self.mjpeg_link(self.nodes.mono_right, self.nodes.xout_right, self.nodes.mono_right.out)
            else:
                self.nodes.mono_right.out.link(self.nodes.xout_right.input)

    def add_nn(self, nn, sync=False, use_depth=False, xout_nn_input=False, xout_sbb=False):
        # TODO adjust this function once passthrough frame type (8) is supported by VideoEncoder (for self.mjpeg_link)
        if xout_nn_input or (sync and self.nn_manager.source == "host"):
            self.nodes.xout_nn_input = self.pipeline.createXLinkOut()
            self.nodes.xout_nn_input.setStreamName(Previews.nn_input.name)
            nn.passthrough.link(self.nodes.xout_nn_input.input)

        if xout_sbb and self.nn_manager.nn_family in ("YOLO", "mobilenet"):
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

            if self.nn_manager.nn_family in ("YOLO", "mobilenet") and use_depth:
                if not hasattr(self.nodes, "xout_depth"):
                    self.nodes.xout_depth = self.pipeline.createXLinkOut()
                    self.nodes.xout_depth.setStreamName(Previews.depth.name)
                nn.passthroughDepth.link(self.nodes.xout_depth.input)

    def create_system_logger(self):
        self.nodes.system_logger = self.pipeline.createSystemLogger()
        self.nodes.system_logger.setRate(1)
        self.nodes.xout_system_logger = self.pipeline.createXLinkOut()
        self.nodes.xout_system_logger.setStreamName("system_logger")
        self.nodes.system_logger.out.link(self.nodes.xout_system_logger.input)

    def create_encoder(self, camera_name, enc_fps):
        """
        Args:
            blob_path (pathlib.Path): Path to the compiled MyriadX blob file
            config_path (pathlib.Path): Path to model config file that is used to download the model
            zoo_name (str): Model name to be taken from model zoo
            zoo_dir (pathlib.Path): Path to model config file that is used to download the model
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
        else:
            raise NotImplementedError("Unable to create encoder for {]".format(camera_name))

        enc = self.pipeline.createVideoEncoder()
        enc.setDefaultProfilePreset(*enc_resolution, enc_fps, enc_profile)
        enc_in.link(enc.input)
        setattr(self.nodes, node_name, enc)

        enc_xout = self.pipeline.createXLinkOut()
        enc.bitstream.link(enc_xout.input)
        enc_xout.setStreamName(xout_name)
        setattr(self.nodes, xout_name, enc_xout)

    def enableLowBandwidth(self):
        self.lowBandwidth = True

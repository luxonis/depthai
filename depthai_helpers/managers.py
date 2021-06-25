import json
import time
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import depthai as dai
import numpy as np

import enum
from depthai_helpers.config_manager import BlobManager
from depthai_helpers.utils import load_module, frame_norm, to_tensor_result


def convert_depth_frame(packet, manager):
    depth_frame = cv2.normalize(packet.getFrame(), None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depth_frame = cv2.equalizeHist(depth_frame)
    depth_frame = cv2.applyColorMap(depth_frame, manager.colorMap)
    return depth_frame


def convert_disparity_frame(packet, manager):
    disparity_frame = (packet.getFrame()*manager.disp_multiplier).astype(np.uint8)
    return disparity_frame


def convert_disparity_to_color(disparity, manager):
    return cv2.applyColorMap(disparity, manager.colorMap)

class Previews(enum.Enum):
    nn_input = partial(lambda packet, _: packet.getCvFrame())
    host = partial(lambda packet, _: packet.getCvFrame())
    color = partial(lambda packet, _: packet.getCvFrame())
    left = partial(lambda packet, _: packet.getCvFrame())
    right = partial(lambda packet, _: packet.getCvFrame())
    rectified_left = partial(lambda packet, _: cv2.flip(packet.getCvFrame(), 1))
    rectified_right = partial(lambda packet, _: cv2.flip(packet.getCvFrame(), 1))
    depth = partial(convert_depth_frame)
    disparity = partial(convert_disparity_frame)
    disparity_color = partial(convert_disparity_to_color)


class PreviewManager:
    def __init__(self, fps, display, colorMap=cv2.COLORMAP_JET, disp_multiplier=255/96):
        self.display = display
        self.frames = {}
        self.raw_frames = {}
        self.fps = fps
        self.colorMap = colorMap
        self.disp_multiplier = disp_multiplier

    def create_queues(self, device):
        def get_output_queue(name):
            cv2.namedWindow(name)
            return device.getOutputQueue(name=name, maxSize=1, blocking=False)
        self.output_queues = [get_output_queue(name) for name in self.display if name != Previews.disparity_color.name]

    def prepare_frames(self, callback):
        for queue in self.output_queues:
            frame = queue.tryGet()
            if frame is not None:
                self.fps.tick(queue.getName())
                frame = getattr(Previews, queue.getName()).value(frame, self)
                callback(frame, queue.getName())
                self.raw_frames[queue.getName()] = frame

                if queue.getName() == Previews.disparity.name:
                    self.fps.tick(Previews.disparity_color.name)
                    self.raw_frames[Previews.disparity_color.name] = Previews.disparity_color.value(frame, self)

            if queue.getName() in self.raw_frames:
                self.frames[queue.getName()] = self.raw_frames[queue.getName()].copy()

                if queue.getName() == Previews.disparity.name:
                    self.frames[Previews.disparity_color.name] = self.raw_frames[Previews.disparity_color.name].copy()

    def show_frames(self, scale=1.0, callback=lambda *a, **k: None):
        for name, frame in self.frames.items():
            if not scale == 1.0:
                h, w, c = frame.shape
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            callback(frame, name)
            cv2.imshow(name, frame)

    def has(self, name):
        return name in self.frames

    def get(self, name):
        return self.frames.get(name, None)


class NNetManager:
    source_choices = ("color", "left", "right", "rectified_left", "rectified_right", "host")
    config = None
    nn_family = None
    handler = None
    labels = None
    input_size = None
    confidence = None
    metadata = None
    openvino_version = None
    output_format = "raw"
    blob_path = None
    count_label = None
    text_color = (255, 255, 255)
    text_type = cv2.FONT_HERSHEY_SIMPLEX
    bbox_color = np.random.random(size=(256, 3)) * 256  # Random Colors for bounding boxes

    def __init__(self, input_size, source, model_dir=None, model_name=None, full_fov=False, flip_detection=False):
        if source not in self.source_choices:
            raise RuntimeError(f"Source {source} is invalid, available {self.source_choices}")

        self.input_size = input_size
        self.full_fov = full_fov
        self.flip_detection = flip_detection
        self.model_name = model_name
        self.model_dir = model_dir
        self.source = source
        self.output_name = f"{self.model_name}_out"
        self.input_name = f"{self.model_name}_in"
        self.blob_manager = BlobManager(model_dir=self.model_dir, model_name=self.model_name)
        # Disaply depth roi bounding boxes

        if model_dir is not None:
            config_path = self.model_dir / Path(self.model_name).with_suffix(f".json")
            if config_path.exists():
                with config_path.open() as f:
                    self.config = json.load(f)
                    if "openvino_version" in self.config:
                        self.openvino_version =getattr(dai.OpenVINO.Version, 'VERSION_' + self.config.get("openvino_version"))
                    nn_config = self.config.get("nn_config", {})
                    self.labels = self.config.get("mappings", {}).get("labels", None)
                    self.nn_family = nn_config.get("NN_family", None)
                    self.output_format = nn_config.get("output_format", "raw")
                    self.metadata = nn_config.get("NN_specific_metadata", {})
                    if "input_size" in nn_config:
                        self.input_size = tuple(map(int, nn_config.get("input_size").split('x')))

                    self.confidence = self.metadata.get("confidence_threshold", nn_config.get("confidence_threshold", None))
                    if 'handler' in self.config:
                        self.handler = load_module(config_path.parent / self.config["handler"])

                        if not callable(getattr(self.handler, "draw", None)) or not callable(getattr(self.handler, "decode", None)):
                            raise RuntimeError("Custom model handler does not contain 'draw' or 'decode' methods!")

        if self.input_size is None:
            raise RuntimeError("Unable to determine the nn input size. Please use --cnn_input_size flag to specify it in WxH format: -nn-size <width>x<height>")

    def normFrame(self, frame):
        if not self.full_fov:
            h = frame.shape[0]
            return np.zeros((h, h))
        else:
            return frame

    def cropOffsetX(self, frame):
        if not self.full_fov:
            h, w = frame.shape[:2]
            return (w - h) // 2
        else:
            return 0

    def create_nn_pipeline(self, p, nodes, shaves=6, use_depth=False, use_sbb=False, minDepth=100, maxDepth=10000, sbbScaleFactor=0.3):
        self.sbb = use_sbb
        if self.nn_family == "mobilenet":
            nn = p.createMobileNetSpatialDetectionNetwork() if use_depth else p.createMobileNetDetectionNetwork()
            nn.setConfidenceThreshold(self.confidence)
        elif self.nn_family == "YOLO":
            nn = p.createYoloSpatialDetectionNetwork() if use_depth else p.createYoloDetectionNetwork()
            nn.setConfidenceThreshold(self.confidence)
            nn.setNumClasses(self.metadata["classes"])
            nn.setCoordinateSize(self.metadata["coordinates"])
            nn.setAnchors(self.metadata["anchors"])
            nn.setAnchorMasks(self.metadata["anchor_masks"])
            nn.setIouThreshold(self.metadata["iou_threshold"])
        else:
            # TODO use createSpatialLocationCalculator
            nn = p.createNeuralNetwork()

        self.blob_path = self.blob_manager.compile(shaves, self.openvino_version)
        nn.setBlobPath(str(self.blob_path))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        nn.input.setQueueSize(2)

        xout = p.createXLinkOut()
        xout.setStreamName(self.output_name)
        nn.out.link(xout.input)
        setattr(nodes, self.model_name, nn)
        setattr(nodes, self.output_name, xout)

        if self.source == "color":
            nodes.cam_rgb.preview.link(nn.input)
        elif self.source == "host":
            xin = p.createXLinkIn()
            xin.setStreamName(self.input_name)
            xin.out.link(nn.input)
            setattr(nodes, self.input_name, xin)
            # Send the video frame back to the host
            nodes.xout_host = p.createXLinkOut()
            nodes.xout_host.setStreamName(Previews.host.name)
            xin.out.link(nodes.xout_host.input)

        elif self.source in ("left", "right", "rectified_left", "rectified_right"):
            nodes.manip = p.createImageManip()
            nodes.manip.initialConfig.setResize(*self.input_size)
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            nodes.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            # NN inputs
            nodes.manip.out.link(nn.input)

            if self.source == "left":
                nodes.mono_left.out.link(nodes.manip.inputImage)
            elif self.source == "right":
                nodes.mono_right.out.link(nodes.manip.inputImage)
            elif self.source == "rectified_left":
                nodes.stereo.rectifiedLeft.link(nodes.manip.inputImage)
            elif self.source == "rectified_right":
                nodes.stereo.rectifiedRight.link(nodes.manip.inputImage)

        if self.nn_family in ("YOLO", "mobilenet") and use_depth:
            nodes.stereo.depth.link(nn.inputDepth)
            nn.setDepthLowerThreshold(minDepth)
            nn.setDepthUpperThreshold(maxDepth)
            nn.setBoundingBoxScaleFactor(sbbScaleFactor)
            nodes.xout_sbb = p.createXLinkOut()
            nodes.xout_sbb.setStreamName("sbb")
            nn.boundingBoxMapping.link(nodes.xout_sbb.input)

        return nn

    def get_label_text(self, label):
        if self.config is None or self.labels is None:
            return label
        elif int(label) < len(self.labels):
            return self.labels[int(label)]
        else:
            print(f"Label of ouf bounds (label_index: {label}, available_labels: {len(self.labels)}")
            return str(label)

    def decode(self, in_nn):
        if self.output_format == "detection":
            detections = in_nn.detections
            if self.flip_detection:
                for detection in detections:
                    # Since rectified frames are horizontally flipped by default
                    swap = detection.xmin
                    detection.xmin = 1 - detection.xmax
                    detection.xmax = 1 - swap
            return detections
        elif self.output_format == "raw":
            if self.handler is not None:
                return self.handler.decode(self, in_nn)
            else:
                try:
                    data = to_tensor_result(in_nn)
                    print("Received NN packet: ", ", ".join([f"{key}: {value.shape}" for key, value in data.items()]))
                except Exception as ex:
                    print("Received NN packet: <Preview unabailable: {}>".format(ex))
        else:
            raise RuntimeError("Unknown output format: {}".format(self.output_format))

    def draw_count(self, source, decoded_data):
        def draw_cnt(frame, cnt):
            cv2.rectangle(frame, (0, 35), (120, 50), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, f"{self.count_label}: {cnt}", (5, 46), self.text_type, 0.5, self.text_color)

        # Count the number of detected objects
        cnt_list = list(filter(lambda x: self.get_label_text(x.label) == self.count_label, decoded_data))
        if isinstance(source, PreviewManager):
            for frame in source.frames.values():
                draw_cnt(frame, len(cnt_list))
        else:
            draw_cnt(source, len(cnt_list))

    def draw(self, source, decoded_data):
        if self.output_format == "detection":
            def draw_detection(frame, detection):
                bbox = frame_norm(self.normFrame(frame), [detection.xmin, detection.ymin, detection.xmax, detection.ymax])
                bbox[::2] += self.cropOffsetX(frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.bbox_color[detection.label], 2)
                cv2.rectangle(frame, (bbox[0], (bbox[1] - 28)), ((bbox[0] + 98), bbox[1]), self.bbox_color[detection.label], cv2.FILLED)
                cv2.putText(frame, self.get_label_text(detection.label), (bbox[0] + 5, bbox[1] - 10),
                            self.text_type, 0.5, (0, 0, 0))
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 58, bbox[1] - 10),
                            self.text_type, 0.5, (0, 0, 0))

                if hasattr(detection, 'spatialCoordinates'):  # Display spatial coordinates as well
                    x_meters = detection.spatialCoordinates.x / 1000
                    y_meters = detection.spatialCoordinates.y / 1000
                    z_meters = detection.spatialCoordinates.z / 1000
                    cv2.putText(frame, "X: {:.2f} m".format(x_meters), (bbox[0] + 10, bbox[1] + 60),
                                self.text_type, 0.5, self.text_color)
                    cv2.putText(frame, "Y: {:.2f} m".format(y_meters), (bbox[0] + 10, bbox[1] + 75),
                                self.text_type, 0.5, self.text_color)
                    cv2.putText(frame, "Z: {:.2f} m".format(z_meters), (bbox[0] + 10, bbox[1] + 90),
                                self.text_type, 0.5, self.text_color)
            for detection in decoded_data:
                if isinstance(source, PreviewManager):
                    for frame in source.frames.values():
                        draw_detection(frame, detection)
                else:
                    draw_detection(source, detection)

            if self.count_label is not None:
                self.draw_count(source, decoded_data)

        elif self.output_format == "raw" and self.handler is not None:
            if isinstance(source, PreviewManager):
                frames = list(source.frames.items())
            else:
                frames = [("host", source)]
            self.handler.draw(self, decoded_data, frames)


class FPSHandler:
    fps_color = (134, 164, 11)
    fps_type = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, cap=None):
        self.timestamp = time.monotonic()
        self.start = None
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self.useCamera = cap is None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if self.start is None:
            self.start = time.monotonic()

        if not self.useCamera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.monotonic()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.monotonic()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            time_diff = time.monotonic() - self.ticks[name]
            return self.ticks_cnt[name] / time_diff if time_diff != 0 else 0
        else:
            return 0

    def fps(self):
        if self.start is None:
            return 0
        time_diff = self.timestamp - self.start
        return self.frame_cnt / time_diff if time_diff != 0 else 0

    def draw_fps(self, source):
        def draw(frame, name: str):
            frame_fps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
            cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, frame_fps, (5, 15), self.fps_type, 0.4, self.fps_color)

            cv2.putText(frame, f"NN FPS:  {round(self.tick_fps('nn'), 1)}", (5, 30), self.fps_type, 0.5, self.fps_color)
        if isinstance(source, PreviewManager):
            for name, frame in source.frames.items():
                draw(frame, name)
        else:
            draw(source, "host")


class PipelineManager:
    def __init__(self, openvino_version=None):
        self.p = dai.Pipeline()
        self.openvino_version=openvino_version
        if openvino_version is not None:
            self.p.setOpenVINOVersion(openvino_version)
        self.nodes = SimpleNamespace()

    def set_nn_manager(self, nn_manager):
        self.nn_manager = nn_manager
        if self.openvino_version is None and self.nn_manager.openvino_version:
            self.p.setOpenVINOVersion(self.nn_manager.openvino_version)
        else:
            self.nn_manager.openvino_version = self.p.getOpenVINOVersion()

    def create_default_queues(self, device):
        for xout in filter(lambda node: isinstance(node, dai.XLinkOut), vars(self.nodes).values()):
            device.getOutputQueue(xout.getStreamName(), maxSize=1, blocking=False)

    def create_color_cam(self, res, fps, full_fov, use_hq):
        # Define a source - color camera
        self.nodes.cam_rgb = self.p.createColorCamera()
        self.nodes.cam_rgb.setPreviewSize(*self.nn_manager.input_size)
        self.nodes.cam_rgb.setInterleaved(False)
        self.nodes.cam_rgb.setResolution(res)
        self.nodes.cam_rgb.setFps(fps)
        self.nodes.cam_rgb.setPreviewKeepAspectRatio(not full_fov)
        self.nodes.xout_rgb = self.p.createXLinkOut()
        self.nodes.xout_rgb.setStreamName(Previews.color.name)
        if use_hq:
            self.nodes.cam_rgb.video.link(self.nodes.xout_rgb.input)
        else:
            self.nodes.cam_rgb.preview.link(self.nodes.xout_rgb.input)

    def create_depth(self, dct, median, lr, extended, subpixel):
        self.nodes.stereo = self.p.createStereoDepth()
        self.nodes.stereo.setConfidenceThreshold(dct)
        self.nodes.stereo.setMedianFilter(median)
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

        self.nodes.xout_depth = self.p.createXLinkOut()
        self.nodes.xout_depth.setStreamName(Previews.depth.name)
        self.nodes.stereo.depth.link(self.nodes.xout_depth.input)

        self.nodes.xout_disparity = self.p.createXLinkOut()
        self.nodes.xout_disparity.setStreamName(Previews.disparity.name)
        self.nodes.stereo.disparity.link(self.nodes.xout_disparity.input)

        self.nodes.xout_rect_left = self.p.createXLinkOut()
        self.nodes.xout_rect_left.setStreamName(Previews.rectified_left.name)
        self.nodes.stereo.rectifiedLeft.link(self.nodes.xout_rect_left.input)

        self.nodes.xout_rect_right = self.p.createXLinkOut()
        self.nodes.xout_rect_right.setStreamName(Previews.rectified_right.name)
        self.nodes.stereo.rectifiedRight.link(self.nodes.xout_rect_right.input)

    def create_left_cam(self, res, fps):
        self.nodes.mono_left = self.p.createMonoCamera()
        self.nodes.mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.nodes.mono_left.setResolution(res)
        self.nodes.mono_left.setFps(fps)

        self.nodes.xout_left = self.p.createXLinkOut()
        self.nodes.xout_left.setStreamName(Previews.left.name)
        self.nodes.mono_left.out.link(self.nodes.xout_left.input)

    def create_right_cam(self, res, fps):
        self.nodes.mono_right = self.p.createMonoCamera()
        self.nodes.mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.nodes.mono_right.setResolution(res)
        self.nodes.mono_right.setFps(fps)

        self.nodes.xout_right = self.p.createXLinkOut()
        self.nodes.xout_right.setStreamName(Previews.right.name)
        self.nodes.mono_right.out.link(self.nodes.xout_right.input)

    def create_nn(self, nn, sync):
        self.nodes.xout_nn_input = self.p.createXLinkOut()
        self.nodes.xout_nn_input.setStreamName(Previews.nn_input.name)
        nn.passthrough.link(self.nodes.xout_nn_input.input)

        if sync:
            if self.nn_manager.source == "color" and hasattr(self.nodes, "xout_rgb"):
                self.nodes.cam_rgb.video.unlink(self.nodes.xout_rgb.input)
                self.nodes.cam_rgb.preview.unlink(self.nodes.xout_rgb.input)
                nn.passthrough.link(self.nodes.xout_rgb.input)
            elif self.nn_manager.source == "host" and hasattr(self.nodes, "xout_host"):
                getattr(self.nodes, self.nn_manager.input_name).out.unlink(self.nodes.xout_host.input)
                nn.passthrough.link(self.nodes.xout_host.input)
            elif self.nn_manager.source == "left" and hasattr(self.nodes, "left"):
                self.nodes.mono_left.out.unlink(self.nodes.xout_left.input)
                nn.passthrough.link(self.nodes.xout_left.input)
            elif self.nn_manager.source == "right" and hasattr(self.nodes, "right"):
                self.nodes.mono_right.out.unlink(self.nodes.xout_right.input)
                nn.passthrough.link(self.nodes.xout_right.input)
            elif self.nn_manager.source == "rectified_left" and hasattr(self.nodes, "rectified_left"):
                self.nodes.stereo.rectifiedLeft.unlink(self.nodes.xout_rect_left.input)
                nn.passthrough.link(self.nodes.xout_rect_left.input)
            elif self.nn_manager.source == "rectified_right" and hasattr(self.nodes, "rectified_right"):
                self.nodes.stereo.rectifiedRight.unlink(self.nodes.xout_rect_right.input)
                nn.passthrough.link(self.nodes.xout_rect_right.input)

            if hasattr(self.nodes, 'xout_depth'):
                self.nodes.stereo.depth.unlink(self.nodes.xout_depth.input)
                nn.passthroughDepth.link(self.nodes.xout_depth.input)

    def create_system_logger(self):
        self.nodes.system_logger = self.p.createSystemLogger()
        self.nodes.system_logger.setRate(1)
        self.nodes.xout_system_logger = self.p.createXLinkOut()
        self.nodes.xout_system_logger.setStreamName("system_logger")
        self.nodes.system_logger.out.link(self.nodes.xout_system_logger.input)


class EncodingManager:
    def __init__(self, pm, encode_config: dict, encode_output=None):
        self.encoding_queues = {}
        self.encoding_nodes = {}
        self.encoding_files = {}
        self.encode_config = encode_config
        self.encode_output = Path(encode_output) or Path(__file__).parent
        self.pm = pm
        for camera_name, enc_fps in self.encode_config.items():
            self.create_encoder(camera_name, enc_fps)
            self.encoding_nodes[camera_name] = getattr(pm.nodes, camera_name + "_enc")

    def create_encoder(self, camera_name, enc_fps):
        allowed_sources = [Previews.left.name, Previews.right.name, Previews.color.name]
        if camera_name not in allowed_sources:
            raise ValueError("Camera param invalid, received {}, available choices: {}".format(camera_name, allowed_sources))
        node_name = camera_name.lower() + '_enc'
        xout_name = node_name + "_xout"
        enc_profile = dai.VideoEncoderProperties.Profile.H264_MAIN

        if camera_name == Previews.color.name:
            if not hasattr(self.pm.nodes, 'cam_rgb'):
                raise RuntimeError("RGB camera not initialized. Call create_color_cam(res, fps) first!")
            enc_resolution = (self.pm.nodes.cam_rgb.getResolutionWidth(), self.pm.nodes.pm.cam_rgb.getResolutionHeight())
            enc_profile = dai.VideoEncoderProperties.Profile.H265_MAIN
            enc_in = self.pm.nodes.cam_rgb.video

        elif camera_name == Previews.left.name:
            if not hasattr(self.pm.nodes, 'mono_left'):
                raise RuntimeError("Left mono camera not initialized. Call create_left_cam(res, fps) first!")
            enc_resolution = (self.pm.nodes.mono_left.getResolutionWidth(), self.pm.nodes.mono_left.getResolutionHeight())
            enc_in = self.pm.nodes.mono_left.out
        elif camera_name == Previews.right.name:
            if not hasattr(self.pm.nodes, 'mono_right'):
                raise RuntimeError("Right mono camera not initialized. Call create_right_cam(res, fps) first!")
            enc_resolution = (self.pm.nodes.mono_right.getResolutionWidth(), self.pm.nodes.mono_right.getResolutionHeight())
            enc_in = self.pm.nodes.mono_right.out
        else:
            raise NotImplementedError("Unable to create encoder for {]".format(camera_name))

        enc = self.pm.p.createVideoEncoder()
        enc.setDefaultProfilePreset(*enc_resolution, enc_fps, enc_profile)
        enc_in.link(enc.input)
        setattr(self.pm.nodes, node_name, enc)

        enc_xout = self.pm.p.createXLinkOut()
        enc.bitstream.link(enc_xout.input)
        enc_xout.setStreamName(xout_name)
        setattr(self.pm.nodes, xout_name, enc_xout)

    def create_default_queues(self, device):
        for camera_name, enc_fps in self.encode_config.items():
            self.encoding_queues[camera_name] = device.getOutputQueue(camera_name + "_enc_xout", maxSize=30, blocking=True)
            self.encoding_files[camera_name] = (self.encode_output / camera_name).with_suffix(
                    ".h265" if self.encoding_nodes[camera_name].getProfile() == dai.VideoEncoderProperties.Profile.H265_MAIN else ".h264"
                ).open('wb')

    def parse_queues(self):
        for name, queue in self.encoding_queues.items():
            while queue.has():
                queue.get().getData().tofile(self.encoding_files[name])

    def close(self):
        def print_manual():
            print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
            cmd = "ffmpeg -framerate {} -i {} -c copy {}"
            for name, file in self.encoding_files.items():
                print(cmd.format(self.encoding_nodes[name].getFrameRate(), file.name, str(Path(file.name).with_suffix('.mp4'))))

        for name, file in self.encoding_files.items():
            file.close()
        try:
            import ffmpy3
            for name, file in self.encoding_files.items():
                fps = self.encoding_nodes[name].getFrameRate()
                out_name = str(Path(file.name).with_suffix('.mp4'))
                try:
                    ff = ffmpy3.FFmpeg(
                        inputs={file.name: "-y"},
                        outputs={out_name: "-c copy -framerate {}".format(fps)}
                    )
                    print("Running conversion command... [{}]".format(ff.cmd))
                    ff.run()
                except ffmpy3.FFExecutableNotFoundError:
                    print("FFMPEG executable not found!")
                    traceback.print_exc()
                    print_manual()
                except ffmpy3.FFRuntimeError:
                    print("FFMPEG runtime error!")
                    traceback.print_exc()
                    print_manual()
            print("Video conversion complete!")
            for name, file in self.encoding_files.items():
                print("Produced file: {}".format(str(Path(file.name).with_suffix('.mp4'))))
        except ImportError:
            print("Module ffmpy3 not fouund!")
            traceback.print_exc()
            print_manual()
        except:
            print("Unknown error!")
            traceback.print_exc()
            print_manual()
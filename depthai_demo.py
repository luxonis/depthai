#!/usr/bin/env python3
import argparse
import enum
import json
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import cv2
import depthai as dai
import numpy as np
from depthai_helpers.version_check import check_depthai_version
import platform

from depthai_helpers.arg_manager import parse_args
from depthai_helpers.config_manager import BlobManager, ConfigManager
from depthai_helpers.utils import frame_norm, to_planar, to_tensor_result, load_module

print('Using depthai module from: ', dai.__file__)
print('Depthai version installed: ', dai.__version__)
if platform.machine() not in ['armv6l', 'aarch64']:
    check_depthai_version()

conf = ConfigManager(parse_args())
conf.linuxCheckApplyUsbRules()
if not conf.useCamera and str(conf.args.video).startswith('https'):
    conf.downloadYTVideo()
conf.adjustPreviewToOptions()

callbacks = load_module(conf.args.callback)
rgb_res = conf.getRgbResolution()
mono_res = conf.getMonoResolution()
bbox_color = np.random.random(size=(256, 3)) * 256  # Random Colors for bounding boxes
text_color = (255, 255, 255)
fps_color = (134, 164, 11)
fps_type = cv2.FONT_HERSHEY_SIMPLEX
text_type = cv2.FONT_HERSHEY_SIMPLEX


def convert_depth_frame(packet):
    depth_frame = cv2.normalize(packet.getFrame(), None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depth_frame = cv2.equalizeHist(depth_frame)
    depth_frame = cv2.applyColorMap(depth_frame, conf.getColorMap())
    return depth_frame


disp_multiplier = 255 / conf.maxDisparity


def convert_disparity_frame(packet):
    disparity_frame = (packet.getFrame()*disp_multiplier).astype(np.uint8)
    return disparity_frame


def convert_disparity_to_color(disparity):
    return cv2.applyColorMap(disparity, conf.getColorMap())


class Previews(enum.Enum):
    nn_input = partial(lambda packet: packet.getCvFrame())
    host = partial(lambda packet: packet.getCvFrame())
    color = partial(lambda packet: packet.getCvFrame())
    left = partial(lambda packet: packet.getCvFrame())
    right = partial(lambda packet: packet.getCvFrame())
    rectified_left = partial(lambda packet: cv2.flip(packet.getCvFrame(), 1))
    rectified_right = partial(lambda packet: cv2.flip(packet.getCvFrame(), 1))
    depth = partial(convert_depth_frame)
    disparity = partial(convert_disparity_frame)
    disparity_color = partial(convert_disparity_to_color)


class PreviewManager:
    def __init__(self, fps):
        self.frames = {}
        self.raw_frames = {}
        self.fps = fps

    def create_queues(self, device):
        def get_output_queue(name):
            cv2.namedWindow(name)
            return device.getOutputQueue(name=name, maxSize=1, blocking=False)
        self.output_queues = [get_output_queue(name) for name in conf.args.show if name != Previews.disparity_color.name]

    def prepare_frames(self):
        for queue in self.output_queues:
            frame = queue.tryGet()
            if frame is not None:
                fps.tick(queue.getName())
                frame = getattr(Previews, queue.getName()).value(frame)
                callbacks.on_new_frame(frame, queue.getName())
                self.raw_frames[queue.getName()] = frame

                if queue.getName() == Previews.disparity.name:
                    fps.tick(Previews.disparity_color.name)
                    self.raw_frames[Previews.disparity_color.name] = Previews.disparity_color.value(frame)

            if queue.getName() in self.raw_frames:
                self.frames[queue.getName()] = self.raw_frames[queue.getName()].copy()

                if queue.getName() == Previews.disparity.name:
                    self.frames[Previews.disparity_color.name] = self.raw_frames[Previews.disparity_color.name].copy()

    def show_frames(self):
        for name, frame in self.frames.items():
            if not conf.args.scale == 1.0:
                h, w, c = frame.shape
                frame = cv2.resize(frame, (int(w * conf.args.scale), int(h * conf.args.scale)), interpolation=cv2.INTER_AREA)
            callbacks.on_show_frame(frame, name)
            cv2.imshow(name, frame)

    def has(self, name):
        return name in self.frames

    def get(self, name):
        return self.frames.get(name, None)


if conf.args.report_file:
    report_file_p = Path(conf.args.report_file).with_suffix('.csv')
    report_file_p.parent.mkdir(parents=True, exist_ok=True)
    report_file = open(conf.args.report_file, 'a')

def print_sys_info(info):
    m = 1024 * 1024 # MiB
    if not conf.args.report_file:
        if "memory" in conf.args.report:
            print(f"Drr used / total - {info.ddrMemoryUsage.used / m:.2f} / {info.ddrMemoryUsage.total / m:.2f} MiB")
            print(f"Cmx used / total - {info.cmxMemoryUsage.used / m:.2f} / {info.cmxMemoryUsage.total / m:.2f} MiB")
            print(f"LeonCss heap used / total - {info.leonCssMemoryUsage.used / m:.2f} / {info.leonCssMemoryUsage.total / m:.2f} MiB")
            print(f"LeonMss heap used / total - {info.leonMssMemoryUsage.used / m:.2f} / {info.leonMssMemoryUsage.total / m:.2f} MiB")
        if "temp" in conf.args.report:
            t = info.chipTemperature
            print(f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa0: {t.upa:.2f}, upa1: {t.dss:.2f}")
        if "cpu" in conf.args.report:
            print(f"Cpu usage - Leon OS: {info.leonCssCpuUsage.average * 100:.2f}%, Leon RT: {info.leonMssCpuUsage.average * 100:.2f} %")
        print("----------------------------------------")
    else:
        data = {}
        if "memory" in conf.args.report:
            data = {
                **data,
                "ddr_used": info.ddrMemoryUsage.used,
                "ddr_total": info.ddrMemoryUsage.total,
                "cmx_used": info.cmxMemoryUsage.used,
                "cmx_total": info.cmxMemoryUsage.total,
                "leon_css_used": info.leonCssMemoryUsage.used,
                "leon_css_total": info.leonCssMemoryUsage.total,
                "leon_mss_used": info.leonMssMemoryUsage.used,
                "leon_mss_total": info.leonMssMemoryUsage.total,
            }
        if "temp" in conf.args.report:
            data = {
                **data,
                "temp_avg": info.chipTemperature.average,
                "temp_css": info.chipTemperature.css,
                "temp_mss": info.chipTemperature.mss,
                "temp_upa0": info.chipTemperature.upa,
                "temp_upa1": info.chipTemperature.dss,
            }
        if "cpu" in conf.args.report:
            data = {
                **data,
                "cpu_css_avg": info.leonCssCpuUsage.average,
                "cpu_mss_avg": info.leonMssCpuUsage.average,
            }

        if report_file.tell() == 0:
            print(','.join(data.keys()), file=report_file)
        callbacks.on_report(data)
        print(','.join(map(str, data.values())), file=report_file)


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
    sbb = False
    source_camera = None
    blob_path = None

    def __init__(self, source, model_dir=None, model_name=None):
        if source not in self.source_choices:
            raise RuntimeError(f"Source {source} is invalid, available {self.source_choices}")

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

        if conf.args.cnn_input_size is None and self.input_size is None:
            raise RuntimeError("Unable to determine the nn input size. Please use --cnn_input_size flag to specify it in WxH format: -nn-size <width>x<height>")

        if conf.args.cnn_input_size:
            self.input_size = tuple(map(int, conf.args.cnn_input_size.split('x')))

        # Count objects detected on the frame
        self.count_label = conf.getCountLabel(self)

    @property
    def should_flip_detection(self):
        return self.source in ("rectified_left", "rectified_right") and not conf.args.stereo_lr_check

    def normFrame(self, frame):
        if not conf.args.full_fov_nn and conf.useCamera:
            h = frame.shape[0]
            return np.zeros((h, h))
        else:
            return frame

    def cropOffsetX(self, frame):
        if not conf.args.full_fov_nn and conf.useCamera:
            h, w = frame.shape[:2]
            return (w - h) // 2
        else:
            return 0


    def create_nn_pipeline(self, p, nodes, use_depth):
        self.sbb = conf.args.spatial_bounding_box and use_depth
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

        self.blob_path = self.blob_manager.compile(conf.args.shaves, self.openvino_version)
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
            self.source_camera = nodes.cam_rgb
        elif self.source == "host":
            xin = p.createXLinkIn()
            xin.setStreamName(self.input_name)
            xin.out.link(nn.input)
            setattr(nodes, self.input_name, xout)
            # Send the video frame back to the host
            if conf.args.sync:
                nodes.xout_host = p.createXLinkOut()
                nodes.xout_host.setStreamName(Previews.host.name)

        elif self.source in ("left", "right", "rectified_left", "rectified_right"):
            nodes.manip = p.createImageManip()
            nodes.manip.initialConfig.setResize(*self.input_size)
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            nodes.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            # NN inputs
            nodes.manip.out.link(nn.input)

            if self.source == "left":
                self.source_camera = nodes.cam_rgb
                nodes.mono_left.out.link(nodes.manip.inputImage)
            elif self.source == "right":
                nodes.mono_right.out.link(nodes.manip.inputImage)
                self.source_camera = nodes.cam_rgb
            elif self.source == "rectified_left":
                nodes.stereo.rectifiedLeft.link(nodes.manip.inputImage)
            elif self.source == "rectified_right":
                nodes.stereo.rectifiedRight.link(nodes.manip.inputImage)

        if self.nn_family in ("YOLO", "mobilenet"):
            if use_depth:
                nodes.stereo.depth.link(nn.inputDepth)
                nn.setDepthLowerThreshold(conf.args.min_depth)
                nn.setDepthUpperThreshold(conf.args.max_depth)

                if conf.args.sbb_scale_factor:
                    nn.setBoundingBoxScaleFactor(conf.args.sbb_scale_factor)

                if self.sbb:
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
            return in_nn.detections
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

    def draw(self, source, decoded_data):
        if self.output_format == "detection":
            def draw_detection(frame, detection):
                bbox = frame_norm(self.normFrame(frame), [detection.xmin, detection.ymin, detection.xmax, detection.ymax])
                bbox[::2] += self.cropOffsetX(frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color[detection.label], 2)
                cv2.rectangle(frame, (bbox[0], (bbox[1] - 28)), ((bbox[0] + 98), bbox[1]), bbox_color[detection.label], cv2.FILLED)
                cv2.putText(frame, self.get_label_text(detection.label), (bbox[0] + 5, bbox[1] - 10),
                            text_type, 0.5, (0, 0, 0))
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 58, bbox[1] - 10),
                            text_type, 0.5, (0, 0, 0))

                if conf.useDepth:  # Display spatial coordinates as well
                    x_meters = detection.spatialCoordinates.x / 1000
                    y_meters = detection.spatialCoordinates.y / 1000
                    z_meters = detection.spatialCoordinates.z / 1000
                    cv2.putText(frame, "X: {:.2f} m".format(x_meters), (bbox[0] + 10, bbox[1] + 60),
                                text_type, 0.5, text_color)
                    cv2.putText(frame, "Y: {:.2f} m".format(y_meters), (bbox[0] + 10, bbox[1] + 75),
                                text_type, 0.5, text_color)
                    cv2.putText(frame, "Z: {:.2f} m".format(z_meters), (bbox[0] + 10, bbox[1] + 90),
                                text_type, 0.5, text_color)
            for detection in decoded_data:
                if self.should_flip_detection:
                    # Since rectified frames are horizontally flipped by default
                    swap = detection.xmin
                    detection.xmin = 1 - detection.xmax
                    detection.xmax = 1 - swap
                if isinstance(source, PreviewManager):
                    for frame in pv.frames.values():
                        draw_detection(frame, detection)
                else:
                    draw_detection(source, detection)

            if self.count_label is not None:
                def draw_cnt(frame, cnt):
                    cv2.rectangle(frame, (0, 35), (120, 50), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, f"{self.count_label}: {cnt}", (5, 46), fps_type, 0.5, fps_color)

                # Count the number of detected objects
                cnt_list = list(filter(lambda x: self.get_label_text(x.label) == self.count_label, decoded_data))
                if isinstance(source, PreviewManager):
                    for frame in pv.frames.values():
                        draw_cnt(frame, len(cnt_list))
                else:
                    draw_cnt(source, len(cnt_list))

        elif self.output_format == "raw" and self.handler is not None:
            if isinstance(source, PreviewManager):
                frames = list(pv.frames.items())
            else:
                frames = [("host", source)]
            self.handler.draw(self, decoded_data, frames)


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.monotonic()
        self.start = None
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if self.start is None:
            self.start = time.monotonic()

        if not conf.useCamera:
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
            frame_fps = f"{name.upper()} FPS: {round(fps.tick_fps(name), 1)}"
            cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, frame_fps, (5, 15), fps_type, 0.4, fps_color)

            cv2.putText(frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), fps_type, 0.5, fps_color)
        if isinstance(source, PreviewManager):
            for name, frame in pv.frames.items():
                draw(frame, name)
        else:
            draw(source, "host")


class PipelineManager:
    def __init__(self):
        self.p = dai.Pipeline()
        if conf.args.openvino_version:
            self.p.setOpenVINOVersion(getattr(dai.OpenVINO.Version, 'VERSION_' + conf.args.openvino_version))
        self.nodes = SimpleNamespace()

    def set_nn_manager(self, nn_manager):
        self.nn_manager = nn_manager
        if not conf.args.openvino_version and self.nn_manager.openvino_version:
            self.p.setOpenVINOVersion(self.nn_manager.openvino_version)
        else:
            self.nn_manager.openvino_version = self.p.getOpenVINOVersion()

    def create_default_queues(self, device):
        for xout in filter(lambda node: isinstance(node, dai.XLinkOut), vars(self.nodes).values()):
            device.getOutputQueue(xout.getStreamName(), maxSize=1, blocking=False)

    def create_color_cam(self, use_hq):
        # Define a source - color camera
        self.nodes.cam_rgb = self.p.createColorCamera()
        self.nodes.cam_rgb.setPreviewSize(*self.nn_manager.input_size)
        self.nodes.cam_rgb.setInterleaved(False)
        self.nodes.cam_rgb.setResolution(rgb_res)
        self.nodes.cam_rgb.setFps(conf.args.rgb_fps)
        self.nodes.cam_rgb.setPreviewKeepAspectRatio(not conf.args.full_fov_nn)
        if Previews.color.name in conf.args.show:
            self.nodes.xout_rgb = self.p.createXLinkOut()
            self.nodes.xout_rgb.setStreamName(Previews.color.name)
            if not conf.args.sync:
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
        if not hasattr(self.nodes, 'mono_left'): self.create_left_cam()
        if not hasattr(self.nodes, 'mono_right'): self.create_right_cam()

        self.nodes.mono_left.out.link(self.nodes.stereo.left)
        self.nodes.mono_right.out.link(self.nodes.stereo.right)

        if Previews.depth.name in conf.args.show:
            self.nodes.xout_depth = self.p.createXLinkOut()
            self.nodes.xout_depth.setStreamName(Previews.depth.name)
            if not conf.args.sync:
                self.nodes.stereo.depth.link(self.nodes.xout_depth.input)
        if Previews.disparity.name in conf.args.show:
            self.nodes.xout_disparity = self.p.createXLinkOut()
            self.nodes.xout_disparity.setStreamName(Previews.disparity.name)
            self.nodes.stereo.disparity.link(self.nodes.xout_disparity.input)
        if Previews.rectified_left.name in conf.args.show:
            self.nodes.xout_rect_left = self.p.createXLinkOut()
            self.nodes.xout_rect_left.setStreamName(Previews.rectified_left.name)
            if not conf.args.sync:
                self.nodes.stereo.rectifiedLeft.link(self.nodes.xout_rect_left.input)
        if Previews.rectified_right.name in conf.args.show:
            self.nodes.xout_rect_right = self.p.createXLinkOut()
            self.nodes.xout_rect_right.setStreamName(Previews.rectified_right.name)
            if not conf.args.sync:
                self.nodes.stereo.rectifiedRight.link(self.nodes.xout_rect_right.input)

    def create_left_cam(self):
        self.nodes.mono_left = self.p.createMonoCamera()
        self.nodes.mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.nodes.mono_left.setResolution(mono_res)
        self.nodes.mono_left.setFps(conf.args.mono_fps)

        if Previews.left.name in conf.args.show:
            self.nodes.xout_left = self.p.createXLinkOut()
            self.nodes.xout_left.setStreamName(Previews.left.name)
            self.nodes.mono_left.out.link(self.nodes.xout_left.input)

    def create_right_cam(self):
        self.nodes.mono_right = self.p.createMonoCamera()
        self.nodes.mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.nodes.mono_right.setResolution(mono_res)
        self.nodes.mono_right.setFps(conf.args.mono_fps)

        if Previews.right.name in conf.args.show:
            self.nodes.xout_right = self.p.createXLinkOut()
            self.nodes.xout_right.setStreamName(Previews.right.name)
            self.nodes.mono_right.out.link(self.nodes.xout_right.input)

    def create_nn(self):
        if callable(self.nn_manager.create_nn_pipeline):
            nn = self.nn_manager.create_nn_pipeline(self.p, self.nodes, use_depth=conf.useDepth)

            if Previews.nn_input.name in conf.args.show:
                self.nodes.xout_nn_input = self.p.createXLinkOut()
                self.nodes.xout_nn_input.setStreamName(Previews.nn_input.name)
                nn.passthrough.link(self.nodes.xout_nn_input.input)

            if conf.args.sync:
                if self.nn_manager.source == "color" and hasattr(self.nodes, "xout_rgb"):
                    nn.passthrough.link(self.nodes.xout_rgb.input)
                elif self.nn_manager.source == "host" and hasattr(self.nodes, "xout_host"):
                    nn.passthrough.link(self.nodes.xout_host.input)
                elif self.nn_manager.source == "left" and hasattr(self.nodes, "left"):
                    nn.passthrough.link(self.nodes.xout_left.input)
                elif self.nn_manager.source == "right" and hasattr(self.nodes, "right"):
                    nn.passthrough.link(self.nodes.xout_right.input)
                elif self.nn_manager.source == "rectified_left" and hasattr(self.nodes, "rectified_left"):
                    nn.passthrough.link(self.nodes.xout_rect_left.input)
                elif self.nn_manager.source == "rectified_right" and hasattr(self.nodes, "rectified_right"):
                    nn.passthrough.link(self.nodes.xout_rect_right.input)

                if hasattr(self.nodes, 'xout_depth'):
                    nn.passthroughDepth.link(self.nodes.xout_depth.input)

    def create_system_logger(self):
        self.nodes.system_logger = self.p.createSystemLogger()
        self.nodes.system_logger.setRate(1)

        if len(conf.args.report) > 0:
            self.nodes.xout_system_logger = self.p.createXLinkOut()
            self.nodes.xout_system_logger.setStreamName("system_logger")
            self.nodes.system_logger.out.link(self.nodes.xout_system_logger.input)


device_info = conf.getDeviceInfo()
pm = PipelineManager()
nn_manager = NNetManager(
    model_name=conf.getModelName(),
    model_dir=conf.getModelDir(),
    source=conf.getModelSource(),
)
pm.set_nn_manager(nn_manager)

# Pipeline is defined, now we can connect to the device
with dai.Device(pm.p.getOpenVINOVersion(), device_info, usb2Mode=conf.args.usb_speed == "usb2") as device:
    conf.adjustParamsToDevice(device)
    cap = cv2.VideoCapture(conf.args.video) if not conf.useCamera else None
    fps = FPSHandler() if conf.useCamera else FPSHandler(cap)

    if conf.useCamera or conf.args.sync:
        pv = PreviewManager(fps)

        if conf.args.camera == "left":
            pm.create_left_cam()
        elif conf.args.camera == "right":
            pm.create_right_cam()
        elif conf.args.camera == "color":
            pm.create_color_cam(conf.useHQ)

        if conf.useDepth:
            pm.create_depth(
                conf.args.disparity_confidence_threshold,
                conf.getMedianFilter(),
                conf.args.stereo_lr_check,
                conf.args.extended_disparity,
                conf.args.subpixel,
            )

    if len(conf.args.report) > 0:
        pm.create_system_logger()

    pm.create_nn()

    # Start pipeline
    device.startPipeline(pm.p)
    pm.create_default_queues(device)
    nn_in = device.getInputQueue(nn_manager.input_name, maxSize=1, blocking=False) if not conf.useCamera else None
    nn_out = device.getOutputQueue(nn_manager.output_name, maxSize=1, blocking=False)

    sbb_out = device.getOutputQueue("sbb", maxSize=1, blocking=False) if nn_manager.sbb else None
    log_out = device.getOutputQueue("system_logger", maxSize=30, blocking=False) if len(conf.args.report) > 0 else None

    if conf.useCamera:
        pv.create_queues(device)
    elif conf.args.sync:
        host_out = device.getOutputQueue(Previews.host.name, maxSize=1, blocking=False)


    seq_num = 0
    host_frame = None
    nn_data = []
    callbacks.on_setup(**locals())

    while True:
        fps.next_iter()
        callbacks.on_iter(**locals())
        if conf.useCamera:
            pv.prepare_frames()

            if sbb_out is not None and pv.has(Previews.depth.name):
                sbb = sbb_out.tryGet()
                sbb_rois = sbb.getConfigData() if sbb is not None else []
                depth_frame = pv.get(Previews.depth.name)
                for roi_data in sbb_rois:
                    roi = roi_data.roi.denormalize(depth_frame.shape[1], depth_frame.shape[0])
                    top_left = roi.topLeft()
                    bottom_right = roi.bottomRight()
                    # Display SBB on the disparity map
                    cv2.rectangle(pv.get("depth"), (int(top_left.x), int(top_left.y)), (int(bottom_right.x), int(bottom_right.y)), bbox_color[0], cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        else:
            read_correctly, host_frame = cap.read()
            if not read_correctly:
                break

            scaled_frame = cv2.resize(host_frame, nn_manager.input_size)
            frame_nn = dai.ImgFrame()
            frame_nn.setSequenceNum(seq_num)
            frame_nn.setType(dai.RawImgFrame.Type.BGR888p)
            frame_nn.setWidth(nn_manager.input_size[0])
            frame_nn.setHeight(nn_manager.input_size[1])
            frame_nn.setData(to_planar(scaled_frame))
            nn_in.send(frame_nn)
            seq_num += 1

            # if high quality, send original frames
            if not conf.useHQ:
                host_frame = scaled_frame
            fps.tick('host')

        in_nn = nn_out.tryGet()
        if in_nn is not None:
            callbacks.on_nn(in_nn)
            if not conf.useCamera and conf.args.sync:
                host_frame = Previews.host.value(host_out.get())
            nn_data = nn_manager.decode(in_nn)
            fps.tick('nn')

        if conf.useCamera:
            nn_manager.draw(pv, nn_data)
            fps.draw_fps(pv)
            pv.show_frames()
        else:
            nn_manager.draw(host_frame, nn_data)
            fps.draw_fps(host_frame)
            cv2.imshow("host", host_frame)

        if log_out:
            logs = log_out.tryGetAll()
            for log in logs:
                print_sys_info(log)

        if cv2.waitKey(1) == ord('q'):
            break

if conf.args.report_file:
    report_file.close()

callbacks.on_teardown(**locals())
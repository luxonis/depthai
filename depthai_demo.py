#!/usr/bin/env python3
import argparse
import enum
import json
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import depthai as dai
import numpy as np

from depthai_helpers.version_check import check_depthai_version
import platform

from depthai_helpers.arg_manager import parse_args
from depthai_helpers.config_manager import BlobManager, ConfigManager
from depthai_helpers.utils import frame_norm, to_planar, to_tensor_result

print('Using depthai module from: ', dai.__file__)
print('Depthai version installed: ', dai.__version__)
if platform.machine() not in ['armv6l', 'aarch64']:
    check_depthai_version()

conf = ConfigManager(parse_args())
conf.linuxCheckApplyUsbRules()

in_w, in_h = conf.getInputSize()
rgb_res = conf.getRgbResolution()
mono_res = conf.getMonoResolution()
median = conf.getMedianFilter()


def convert_depth_frame(packet):
    depth_frame = cv2.normalize(packet.getFrame(), None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depth_frame = cv2.equalizeHist(depth_frame)
    depth_frame = cv2.applyColorMap(depth_frame, conf.getColorMap())
    return depth_frame


disp_multiplier = 255 / conf.args.max_disparity


def convert_disparity_frame(packet):
    disparity_frame = (packet.getFrame()*disp_multiplier).astype(np.uint8)
    return disparity_frame


def convert_disparity_to_color(disparity):
    return cv2.applyColorMap(disparity, conf.getColorMap())


class Previews(enum.Enum):
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
            cv2.imshow(name, frame)
    
    def has(self, name):
        return name in self.frames
    
    def get(self, name):
        return self.frames.get(name, None)
        

class NNetManager:
    source_choices = ("color", "left", "right", "rectified_left", "rectified_right", "host")
    config = None
    nn_family = None
    labels = None
    confidence = None
    metadata = None
    output_format = None
    sbb = False

    def __init__(self, source, use_depth, use_hq, model_dir=None, model_name=None):
        if source not in self.source_choices:
            raise RuntimeError(f"Source {source} is invalid, available {self.source_choices}")

        self.model_name = model_name
        self.model_dir = model_dir
        self.source = source
        self.use_depth = use_depth
        self.use_hq = use_hq
        self.output_name = f"{self.model_name}_out"
        self.input_name = f"{self.model_name}_in"
        self.blob_path = BlobManager(model_dir=self.model_dir, model_name=self.model_name).compile(conf.args.shaves)
        # Disaply depth roi bounding boxes
        self.sbb = conf.args.spatial_bounding_box and self.use_depth

        if model_dir is not None:
            config_path = self.model_dir / Path(self.model_name).with_suffix(f".json")
            if config_path.exists():
                with config_path.open() as f:
                    self.config = json.load(f)
                    nn_config = self.config.get("NN_config", {})
                    self.labels = self.config.get("mappings", {}).get("labels", None)
                    self.nn_family = nn_config.get("NN_family", None)
                    self.output_format = nn_config.get("output_format", None)
                    self.metadata = nn_config.get("NN_specific_metadata", {})

                    self.confidence = self.metadata.get("confidence_threshold", nn_config.get("confidence_threshold", None))

    def create_nn_pipeline(self, p, nodes):
        if self.nn_family == "mobilenet":
            nn = p.createMobileNetSpatialDetectionNetwork() if self.use_depth else p.createMobileNetDetectionNetwork()
            nn.setConfidenceThreshold(self.confidence)
        elif self.nn_family == "YOLO":
            nn = p.createYoloSpatialDetectionNetwork() if self.use_depth else p.createYoloDetectionNetwork()
            nn.setConfidenceThreshold(self.confidence)
            nn.setNumClasses(self.metadata["classes"])
            nn.setCoordinateSize(self.metadata["coordinates"])
            nn.setAnchors(self.metadata["anchors"])
            nn.setAnchorMasks(self.metadata["anchor_masks"])
            nn.setIouThreshold(self.metadata["iou_threshold"])
        else:
            # TODO use createSpatialLocationCalculator
            nn = p.createNeuralNetwork()

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
            setattr(nodes, self.input_name, xout)
        elif self.source in ("left", "right", "rectified_left", "rectified_right"):
            nodes.manip = p.createImageManip()
            nodes.manip.initialConfig.setResize(in_w, in_h)
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

        if self.nn_family in ("YOLO", "mobilenet"):
            if self.use_depth:
                nodes.stereo.depth.link(nn.inputDepth)
                nn.setDepthLowerThreshold(100)
                nn.setDepthUpperThreshold(3000)

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
    
    def draw_detections(self, source, detections):
        def draw_detection(frame, detection):
            bbox = frame_norm(frame, [detection.xmin, detection.ymin, detection.xmax, detection.ymax])
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, self.get_label_text(detection.label), (bbox[0] + 10, bbox[1] + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            if conf.useDepth:  # Display coordinates as well
                cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (bbox[0] + 10, bbox[1] + 60),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (bbox[0] + 10, bbox[1] + 75),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (bbox[0] + 10, bbox[1] + 90),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        for detection in detections:
            if self.source in ("rectified_left", "rectified_right"):
                # Since rectified frames are horizontally flipped by default
                swap = detection.xmin
                detection.xmin = 1 - detection.xmax
                detection.xmax = 1 - swap
            if isinstance(source, PreviewManager):
                for frame in pv.frames.values():
                    draw_detection(frame, detection)
            else:
                draw_detection(source, detection)


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
            cv2.putText(frame, frame_fps, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv2.putText(frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        if isinstance(source, PreviewManager):
            for name, frame in pv.frames.items():
                draw(frame, name)
        else:
            draw(source, "host")


class PipelineManager:
    def __init__(self, nn_manager):
        self.p = dai.Pipeline()
        self.nodes = SimpleNamespace()
        self.nn_manager = nn_manager

    def create_default_queues(self, device):
        for xout in filter(lambda node: isinstance(node, dai.XLinkOut), vars(self.nodes).values()):
            device.getOutputQueue(xout.getStreamName(), maxSize=1, blocking=False)

    def create_color_cam(self, use_hq):
        # Define a source - color camera
        self.nodes.cam_rgb = self.p.createColorCamera()
        self.nodes.cam_rgb.setPreviewSize(in_w, in_h)
        self.nodes.cam_rgb.setInterleaved(False)
        self.nodes.cam_rgb.setResolution(rgb_res)
        self.nodes.cam_rgb.setFps(conf.args.rgb_fps)
        self.nodes.cam_rgb.setPreviewKeepAspectRatio(False)
        if Previews.color.name in conf.args.show:
            self.nodes.xout_rgb = self.p.createXLinkOut()
            self.nodes.xout_rgb.setStreamName(Previews.color.name)
            if not conf.args.sync:
                if use_hq:
                    self.nodes.cam_rgb.video.link(self.nodes.xout_rgb.input)
                else:
                    self.nodes.cam_rgb.preview.link(self.nodes.xout_rgb.input)

    def create_depth(self, dct, median, lr):
        self.nodes.stereo = self.p.createStereoDepth()
        self.nodes.stereo.setConfidenceThreshold(dct)
        self.nodes.stereo.setMedianFilter(median)
        self.nodes.stereo.setLeftRightCheck(lr)

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
            nn = self.nn_manager.create_nn_pipeline(self.p, self.nodes)

            if conf.args.sync:
                if self.nn_manager.source == "color" and hasattr(self.nodes, "xout_rgb"):
                    nn.passthrough.link(self.nodes.xout_rgb.input)
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


device_info = conf.getDeviceInfo()

# Pipeline is defined, now we can connect to the device
with dai.Device(dai.OpenVINO.Version.VERSION_2021_3, device_info) as device:
    conf.adjustParamsToDevice(device)

    nn_manager = NNetManager(
        model_name=conf.getModelName(),
        model_dir=conf.getModelDir(),
        source=conf.getModelSource(),
        use_depth=conf.useDepth,
        use_hq=conf.useHQ
    )

    pm = PipelineManager(nn_manager)
    cap = cv2.VideoCapture(conf.args.video) if not conf.useCamera else None
    fps = FPSHandler() if conf.useCamera else FPSHandler(cap)

    if conf.useCamera:
        pv = PreviewManager(fps)

        if conf.args.camera == "left":
            pm.create_left_cam()
        elif conf.args.camera == "right":
            pm.create_right_cam()
        elif conf.args.camera == "color":
            pm.create_color_cam(conf.useHQ)

        if conf.useDepth:
            pm.create_depth(conf.args.disparity_confidence_threshold, median, conf.args.stereo_lr_check)

    pm.create_nn()

    # Start pipeline
    device.startPipeline(pm.p)
    pm.create_default_queues(device)
    nn_in = device.getInputQueue(nn_manager.input_name, maxSize=1, blocking=False) if not conf.useCamera else None
    nn_out = device.getOutputQueue(nn_manager.output_name, maxSize=1, blocking=False)

    sbb_out = device.getOutputQueue("sbb", maxSize=1, blocking=False) if nn_manager.sbb else None

    if conf.useCamera:
        pv.create_queues(device)
    # cam_out = device.getOutputQueue(name=current_stream.name, maxSize=4, blocking=False) if conf.useCamera else None

    seq_num = 0
    host_frame = None
    detections = []
    # Spatial bounding box ROIs (region of interests)
    color = (255, 255, 255)

    while True:
        fps.next_iter()
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
                    cv2.rectangle(pv.get("depth"), (int(top_left.x), int(top_left.y)), (int(bottom_right.x), int(bottom_right.y)), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        else:
            read_correctly, host_frame = cap.read()
            if not read_correctly:
                break

            scaled_frame = cv2.resize(host_frame, (in_w, in_h))
            frame_nn = dai.ImgFrame()
            frame_nn.setSequenceNum(seq_num)
            frame_nn.setWidth(in_w)
            frame_nn.setHeight(in_h)
            frame_nn.setData(to_planar(scaled_frame))
            nn_in.send(frame_nn)
            seq_num += 1

            # if high quality, send original frames
            if not conf.useHQ:
                host_frame = scaled_frame
            fps.tick('host')

        in_nn = nn_out.tryGetAll()
        if len(in_nn) > 0:
            if nn_manager.output_format == "detection":
                detections = in_nn[-1].detections
            for packet in in_nn:
                if nn_manager.output_format is None:
                    try:
                        print("Received NN packet: ", to_tensor_result(packet))
                    except Exception as ex:
                        print("Received NN packet: <Preview unabailable: {}>".format(ex))
                fps.tick('nn')

        if conf.useCamera:
            nn_manager.draw_detections(pv, detections)
            fps.draw_fps(pv)
            pv.show_frames()
        else:
            nn_manager.draw_detections(host_frame, detections)
            fps.draw_fps(host_frame)
            cv2.imshow("host", host_frame)

        if cv2.waitKey(1) == ord('q'):
            break

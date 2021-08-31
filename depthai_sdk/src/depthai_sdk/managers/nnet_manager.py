import json
from pathlib import Path
import depthai as dai
import cv2
import numpy as np

from .preview_manager import PreviewManager
from ..previews import Previews
from ..utils import load_module, to_tensor_result, frame_norm, to_planar


class NNetManager:
    source_choices = ("color", "left", "right", "rectified_left", "rectified_right", "host")
    flip_detection = False
    full_fov = False
    config = None
    nn_family = None
    handler = None
    labels = None
    input_size = None
    confidence = None
    metadata = None
    openvino_version = None
    output_format = "raw"
    source = None
    count_label = None
    text_bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    line_type = cv2.LINE_AA
    text_type = cv2.FONT_HERSHEY_SIMPLEX
    bbox_color = np.random.random(size=(256, 3)) * 256  # Random Colors for bounding boxes
    input_queue = None
    output_queue = None

    def __init__(self, input_size, blob_manager):

        self.input_size = input_size
        self.blob_manager = blob_manager

    def read_config(self, path):
        config_path = Path(path)
        if not config_path.exists():
            raise ValueError("Path {} does not exist!".format(path))

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

    def normFrame(self, frame):
        if not self.full_fov:
            scale_f = frame.shape[0] / self.input_size[1]
            return np.zeros((int(self.input_size[1] * scale_f), int(self.input_size[0] * scale_f)))
        else:
            return frame

    def cropOffsetX(self, frame):
        if not self.full_fov:
            cropped_w = (frame.shape[0] / self.input_size[1]) * self.input_size[0]
            return int((frame.shape[1] - cropped_w) // 2)
        else:
            return 0

    def create_nn_pipeline(self, p, nodes, source, flip_detection=False, shaves=6, use_depth=False, use_sbb=False, minDepth=100, maxDepth=10000, sbbScaleFactor=0.3, full_fov=False):
        if source not in self.source_choices:
            raise RuntimeError(f"Source {source} is invalid, available {self.source_choices}")
        if self.input_size is None:
            raise RuntimeError("Unable to determine the nn input size. Please use --cnn_input_size flag to specify it in WxH format: -nn-size <width>x<height>")

        self.source = source
        self.flip_detection = flip_detection
        self.full_fov = full_fov
        self.sbb = use_sbb
        if self.nn_family == "mobilenet":
            nodes.nn = p.createMobileNetSpatialDetectionNetwork() if use_depth else p.createMobileNetDetectionNetwork()
            nodes.nn.setConfidenceThreshold(self.confidence)
        elif self.nn_family == "YOLO":
            nodes.nn = p.createYoloSpatialDetectionNetwork() if use_depth else p.createYoloDetectionNetwork()
            nodes.nn.setConfidenceThreshold(self.confidence)
            nodes.nn.setNumClasses(self.metadata["classes"])
            nodes.nn.setCoordinateSize(self.metadata["coordinates"])
            nodes.nn.setAnchors(self.metadata["anchors"])
            nodes.nn.setAnchorMasks(self.metadata["anchor_masks"])
            nodes.nn.setIouThreshold(self.metadata["iou_threshold"])
        else:
            # TODO use createSpatialLocationCalculator
            nodes.nn = p.createNeuralNetwork()

        nodes.nn.setBlobPath(str(self.blob_manager.blob_path))
        nodes.nn.setNumInferenceThreads(2)
        nodes.nn.input.setBlocking(False)
        nodes.nn.input.setQueueSize(2)

        nodes.xout_nn = p.createXLinkOut()
        nodes.xout_nn.setStreamName("nn_out")
        nodes.nn.out.link(nodes.xout_nn.input)

        if self.source == "color":
            nodes.cam_rgb.preview.link(nodes.nn.input)
        elif self.source == "host":
            nodes.xin_nn = p.createXLinkIn()
            nodes.xin_nn.setStreamName("nn_in")
            nodes.xin_nn.out.link(nodes.nn.input)
        elif self.source in ("left", "right", "rectified_left", "rectified_right"):
            nodes.manip = p.createImageManip()
            nodes.manip.initialConfig.setResize(*self.input_size)
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            nodes.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            # NN inputs
            nodes.manip.out.link(nodes.nn.input)
            nodes.manip.setKeepAspectRatio(not self.full_fov)

            if self.source == "left":
                nodes.mono_left.out.link(nodes.manip.inputImage)
            elif self.source == "right":
                nodes.mono_right.out.link(nodes.manip.inputImage)
            elif self.source == "rectified_left":
                nodes.stereo.rectifiedLeft.link(nodes.manip.inputImage)
            elif self.source == "rectified_right":
                nodes.stereo.rectifiedRight.link(nodes.manip.inputImage)

        if self.nn_family in ("YOLO", "mobilenet") and use_depth:
            nodes.stereo.depth.link(nodes.nn.inputDepth)
            nodes.nn.setDepthLowerThreshold(minDepth)
            nodes.nn.setDepthUpperThreshold(maxDepth)
            nodes.nn.setBoundingBoxScaleFactor(sbbScaleFactor)

        return nodes.nn

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
            cv2.putText(frame, f"{self.count_label}: {cnt}", (5, 46), self.text_type, 0.5, self.text_bg_color, 4, self.line_type)
            cv2.putText(frame, f"{self.count_label}: {cnt}", (5, 46), self.text_type, 0.5, self.text_color, 1, self.line_type)

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
                if self.source == Previews.color.name and not self.full_fov:
                    bbox[::2] += self.cropOffsetX(frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.bbox_color[detection.label], 2)
                cv2.rectangle(frame, (bbox[0], (bbox[1] - 28)), ((bbox[0] + 110), bbox[1]), self.bbox_color[detection.label], cv2.FILLED)
                cv2.putText(frame, self.get_label_text(detection.label), (bbox[0] + 5, bbox[1] - 10),
                            self.text_type, 0.5, (0, 0, 0), 1, self.line_type)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 62, bbox[1] - 10),
                            self.text_type, 0.5, (0, 0, 0), 1, self.line_type)

                if hasattr(detection, 'spatialCoordinates'):  # Display spatial coordinates as well
                    x_meters = detection.spatialCoordinates.x / 1000
                    y_meters = detection.spatialCoordinates.y / 1000
                    z_meters = detection.spatialCoordinates.z / 1000
                    cv2.putText(frame, "X: {:.2f} m".format(x_meters), (bbox[0] + 10, bbox[1] + 60),
                                self.text_type, 0.5, self.text_bg_color, 4, self.line_type)
                    cv2.putText(frame, "X: {:.2f} m".format(x_meters), (bbox[0] + 10, bbox[1] + 60),
                                self.text_type, 0.5, self.text_color, 1, self.line_type)
                    cv2.putText(frame, "Y: {:.2f} m".format(y_meters), (bbox[0] + 10, bbox[1] + 75),
                                self.text_type, 0.5, self.text_bg_color, 4, self.line_type)
                    cv2.putText(frame, "Y: {:.2f} m".format(y_meters), (bbox[0] + 10, bbox[1] + 75),
                                self.text_type, 0.5, self.text_color, 1, self.line_type)
                    cv2.putText(frame, "Z: {:.2f} m".format(z_meters), (bbox[0] + 10, bbox[1] + 90),
                                self.text_type, 0.5, self.text_bg_color, 4, self.line_type)
                    cv2.putText(frame, "Z: {:.2f} m".format(z_meters), (bbox[0] + 10, bbox[1] + 90),
                                self.text_type, 0.5, self.text_color, 1, self.line_type)
            for detection in decoded_data:
                if isinstance(source, PreviewManager):
                    for name, frame in source.frames.items():
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

    def createQueues(self, device):
        if self.source == "host":
            self.input_queue = device.getInputQueue("nn_in", maxSize=1, blocking=False)
        self.output_queue = device.getOutputQueue("nn_out", maxSize=1, blocking=False)

    def sendInputFrame(self, frame, seq_num=None):
        if self.input_queue is None:
            raise RuntimeError("Unable to send image, no input queue is present! Call `createQueues(device)` first!")

        scaled_frame = cv2.resize(frame, self.input_size)
        frame_nn = dai.ImgFrame()
        if seq_num is not None:
            frame_nn.setSequenceNum(seq_num)
        frame_nn.setType(dai.ImgFrame.Type.BGR888p)
        frame_nn.setWidth(self.input_size[0])
        frame_nn.setHeight(self.input_size[1])
        frame_nn.setData(to_planar(scaled_frame))
        self.input_queue.send(frame_nn)

        return scaled_frame
import json
from pathlib import Path
import depthai as dai
import cv2
import numpy as np

from .blob_manager import BlobManager
from .preview_manager import PreviewManager
from ..previews import Previews
from ..utils import load_module, to_tensor_result, frame_norm


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
    blob_path = None
    source = None
    count_label = None
    text_bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    line_type = cv2.LINE_AA
    text_type = cv2.FONT_HERSHEY_SIMPLEX
    bbox_color = np.random.random(size=(256, 3)) * 256  # Random Colors for bounding boxes

    def __init__(self, input_size, model_dir=None, model_name=None):

        self.input_size = input_size
        self.model_name = model_name
        self.model_dir = model_dir
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
        self.source = source
        self.flip_detection = flip_detection
        self.full_fov = full_fov
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

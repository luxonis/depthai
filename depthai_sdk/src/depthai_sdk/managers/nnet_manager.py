import json
from pathlib import Path
import depthai as dai
import cv2
import numpy as np

from .preview_manager import PreviewManager
from ..previews import Previews
from ..utils import load_module, to_tensor_result, frame_norm, to_planar


class NNetManager:
    """
    Manager class handling all NN-related functionalities. It's capable of creating appropreate nodes and connections,
    decoding neural network output automatically or by using external handler file.
    """

    def __init__(self, input_size, nn_family=None):
        """
        Args:
            input_size (tuple): Desired NN input size, should match the input size defined in the network itself (width, height)
            nn_family (str): type of NeuralNetwork to be processed. Supported: :code:`"YOLO"` and :code:`mobilenet`
        """
        self.input_size = input_size
        self._nn_family = nn_family
        if nn_family in ("YOLO", "mobilenet"):
            self._output_format = "detection"

    #: list: List of available neural network inputs
    source_choices = ("color", "left", "right", "rectified_left", "rectified_right", "host")
    #: str: Selected neural network input
    source = None
    #: tuple: NN input size (width, height)
    input_size = None
    #: depthai.OpenVINO.Version: OpenVINO version, available only if parsed from config file (see :func:`read_config`)
    openvino_version = None
    #: depthai.DataInputQueue: DepthAI input queue object that allows to send images from host to device (used only with :code:`host` source)
    input_queue = None
    #: depthai.DataOutputQueue: DepthAI output queue object that allows to receive NN results from the device.
    output_queue = None

    _bbox_colors = np.random.random(size=(256, 3)) * 256  # Random Colors for bounding boxes
    _count_label = None
    _text_bg_color = (0, 0, 0)
    _text_color = (255, 255, 255)
    _line_type = cv2.LINE_AA
    _text_type = cv2.FONT_HERSHEY_SIMPLEX
    _output_format = "raw"
    _confidence = 0.7
    _metadata = None
    _flip_detection = False
    _full_fov = True
    _config = None
    _nn_family = None
    _handler = None
    _labels = None

    def read_config(self, path):
        """
        Parses the model config file and adjusts NNetManager values accordingly. It's advised to create a config file
        for every new network, as it allows to use dedicated NN nodes (for `MobilenetSSD <https://github.com/luxonis/depthai/blob/main/resources/nn/mobilenet-ssd/mobilenet-ssd.json>`__ and `YOLO <https://github.com/luxonis/depthai/blob/main/resources/nn/tiny-yolo-v3/tiny-yolo-v3.json>`__)
        or use `custom handler <https://github.com/luxonis/depthai/blob/main/resources/nn/openpose2/openpose2.json>`__ to process and display custom network results

        Args:
            path (pathlib.Path): Path to model config file (.json)

        Raises:
            ValueError: If path to config file does not exist
            RuntimeError: If custom handler does not contain :code:`draw` or :code:`show` methods
        """
        config_path = Path(path)
        if not config_path.exists():
            raise ValueError("Path {} does not exist!".format(path))

        with config_path.open() as f:
            self._config = json.load(f)
            if "openvino_version" in self._config:
                self.openvino_version =getattr(dai.OpenVINO.Version, 'VERSION_' + self._config.get("openvino_version"))
            nn_config = self._config.get("nn_config", {})
            self._labels = self._config.get("mappings", {}).get("labels", None)
            self._nn_family = nn_config.get("NN_family", None)
            self._output_format = nn_config.get("output_format", "raw")
            self._metadata = nn_config.get("NN_specific_metadata", {})
            if "input_size" in nn_config:
                self.input_size = tuple(map(int, nn_config.get("input_size").split('x')))

            self._confidence = self._metadata.get("confidence_threshold", nn_config.get("confidence_threshold", None))
            if 'handler' in self._config:
                self._handler = load_module(config_path.parent / self._config["handler"])

                if not callable(getattr(self._handler, "draw", None)) or not callable(getattr(self._handler, "decode", None)):
                    raise RuntimeError("Custom model handler does not contain 'draw' or 'decode' methods!")


    def _normFrame(self, frame):
        if not self._full_fov:
            scale_f = frame.shape[0] / self.input_size[1]
            return np.zeros((int(self.input_size[1] * scale_f), int(self.input_size[0] * scale_f)))
        else:
            return frame

    def _cropOffsetX(self, frame):
        if not self._full_fov:
            cropped_w = (frame.shape[0] / self.input_size[1]) * self.input_size[0]
            return int((frame.shape[1] - cropped_w) // 2)
        else:
            return 0

    def create_nn_pipeline(self, pipeline, nodes, source, blob_path, flip_detection=False, use_depth=False, minDepth=100, maxDepth=10000, sbbScaleFactor=0.3, full_fov=True):
        """
        Creates nodes and connections in provided pipeline that will allow to run NN model and consume it's results.

        Args:
            pipeline (depthai.Pipeline): Pipeline instance
            nodes (types.SimpleNamespace): Object cointaining all of the nodes added to the pipeline. Available in :attr:`depthai_sdk.managers.PipelineManager.nodes`
            source (str): Neural network input source, one of :attr:`source_choices`
            blob_path (pathlib.Path): Path to MyriadX blob. Might be useful to use together with
                :func:`depthai_sdk.managers.BlobManager.getBlob()` for dynamic blob compilation
            use_depth (bool): If set to True, produced detections will have spatial coordinates included
            minDepth (int): Minimum depth distance in centimeters
            maxDepth (int): Maximum depth distance in centimeters
            sbbScaleFactor (float): Scale of the bounding box that will be used to calculate spatial coordinates for
                detection. If set to 0.3, it will scale down center-wise the bounding box to 0.3 of it's original size
                and use it to calculate spatial location of the object
            full_fov (bool): If set to False, manager will include crop offset when scaling the detections.
                Usually should be set to True (if you don't perform aspect ratio crop or when `keepAspectRatio` flag
                on camera/manip node is set to False
            flip_detection (bool): Whether the bounding box coordinates should be flipped horizontally. Useful when
                using rectified images as input.

        Returns:
            depthai.node.NeuralNetwork: Configured NN node that was added to the pipeline

        Raises:
            RuntimeError: If source is not a valid choice or when input size has not been set.
        """
        if source not in self.source_choices:
            raise RuntimeError(f"Source {source} is invalid, available {self.source_choices}")
        if self.input_size is None:
            raise RuntimeError("Unable to determine the nn input size. Please use --cnn_input_size flag to specify it in WxH format: -nn-size <width>x<height>")

        self.source = source
        self._flip_detection = flip_detection
        self._full_fov = full_fov
        if self._nn_family == "mobilenet":
            nodes.nn = pipeline.createMobileNetSpatialDetectionNetwork() if use_depth else pipeline.createMobileNetDetectionNetwork()
            nodes.nn.setConfidenceThreshold(self._confidence)
        elif self._nn_family == "YOLO":
            nodes.nn = pipeline.createYoloSpatialDetectionNetwork() if use_depth else pipeline.createYoloDetectionNetwork()
            nodes.nn.setConfidenceThreshold(self._confidence)
            nodes.nn.setNumClasses(self._metadata["classes"])
            nodes.nn.setCoordinateSize(self._metadata["coordinates"])
            nodes.nn.setAnchors(self._metadata["anchors"])
            nodes.nn.setAnchorMasks(self._metadata["anchor_masks"])
            nodes.nn.setIouThreshold(self._metadata["iou_threshold"])
        else:
            # TODO use createSpatialLocationCalculator
            nodes.nn = pipeline.createNeuralNetwork()

        nodes.nn.setBlobPath(str(blob_path))
        nodes.nn.setNumInferenceThreads(2)
        nodes.nn.input.setBlocking(False)
        nodes.nn.input.setQueueSize(2)

        nodes.xout_nn = pipeline.createXLinkOut()
        nodes.xout_nn.setStreamName("nn_out")
        nodes.nn.out.link(nodes.xout_nn.input)

        if self.source == "color":
            nodes.cam_rgb.preview.link(nodes.nn.input)
        elif self.source == "host":
            nodes.xin_nn = pipeline.createXLinkIn()
            nodes.xin_nn.setStreamName("nn_in")
            nodes.xin_nn.out.link(nodes.nn.input)
        elif self.source in ("left", "right", "rectified_left", "rectified_right"):
            nodes.manip = pipeline.createImageManip()
            nodes.manip.initialConfig.setResize(*self.input_size)
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            nodes.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            # NN inputs
            nodes.manip.out.link(nodes.nn.input)
            nodes.manip.setKeepAspectRatio(not self._full_fov)

            if self.source == "left":
                nodes.mono_left.out.link(nodes.manip.inputImage)
            elif self.source == "right":
                nodes.mono_right.out.link(nodes.manip.inputImage)
            elif self.source == "rectified_left":
                nodes.stereo.rectifiedLeft.link(nodes.manip.inputImage)
            elif self.source == "rectified_right":
                nodes.stereo.rectifiedRight.link(nodes.manip.inputImage)

        if self._nn_family in ("YOLO", "mobilenet") and use_depth:
            nodes.stereo.depth.link(nodes.nn.inputDepth)
            nodes.nn.setDepthLowerThreshold(minDepth)
            nodes.nn.setDepthUpperThreshold(maxDepth)
            nodes.nn.setBoundingBoxScaleFactor(sbbScaleFactor)

        return nodes.nn

    def get_label_text(self, label):
        """
        Retrieves text assigned to specific label

        Args:
            label (int): Integer representing detection label, usually returned from NN node

        Returns:
            str: Label text assigned to specific label id or label id

        Raises:
            RuntimeError: If source is not a valid choice or when input size has not been set.
        """
        if self._config is None or self._labels is None:
            return str(label)
        elif int(label) < len(self._labels):
            return self._labels[int(label)]
        else:
            print(f"Label of ouf bounds (label_index: {label}, available_labels: {len(self._labels)}")
            return str(label)

    def decode(self, in_nn):
        """
        Decodes NN output. Performs generic handling for supported detection networks or calls custom handler methods

        Args:
            in_nn (depthai.NNData): Integer representing detection label, usually returned from NN node

        Returns:
            Decoded NN data

        Raises:
            RuntimeError: if output_format specified in model config file is not recognized
        """
        if self._output_format == "detection":
            detections = in_nn.detections
            if self._flip_detection:
                for detection in detections:
                    # Since rectified frames are horizontally flipped by default
                    swap = detection.xmin
                    detection.xmin = 1 - detection.xmax
                    detection.xmax = 1 - swap
            return detections
        elif self._output_format == "raw":
            if self._handler is not None:
                return self._handler.decode(self, in_nn)
            else:
                try:
                    data = to_tensor_result(in_nn)
                    print("Received NN packet: ", ", ".join([f"{key}: {value.shape}" for key, value in data.items()]))
                except Exception as ex:
                    print("Received NN packet: <Preview unabailable: {}>".format(ex))
        else:
            raise RuntimeError("Unknown output format: {}".format(self._output_format))

    def _draw_count(self, source, decoded_data):
        def draw_cnt(frame, cnt):
            cv2.putText(frame, f"{self._count_label}: {cnt}", (5, 46), self._text_type, 0.5, self._text_bg_color, 4, self._line_type)
            cv2.putText(frame, f"{self._count_label}: {cnt}", (5, 46), self._text_type, 0.5, self._text_color, 1, self._line_type)

        # Count the number of detected objects
        cnt_list = list(filter(lambda x: self.get_label_text(x.label) == self._count_label, decoded_data))
        if isinstance(source, PreviewManager):
            for frame in source.frames.values():
                draw_cnt(frame, len(cnt_list))
        else:
            draw_cnt(source, len(cnt_list))

    def draw(self, source, decoded_data):
        """
        Draws NN results onto the frames. It's responsible to correctly map the results onto each frame requested,
        including applying crop offset or preparing a correct normalization frame, then draws them with all information
        provided (confidence, label, spatial location, label count).

        Also, it's able to call custom nn handler method :code:`draw` to hand over drawing the results

        Args:
            source (depthai_sdk.managers.PreviewManager | numpy.ndarray): Draw target.
                If supplied with a regular frame, it will draw the count on that frame

                If supplied with :class:`depthai_sdk.managers.PreviewManager` instance, it will print the count label
                on all of the frames that it stores

            decoded_data: Detections from neural network node, usually returned from :func:`decode` method
        """
        if self._output_format == "detection":
            def draw_detection(frame, detection):
                bbox = frame_norm(self._normFrame(frame), [detection.xmin, detection.ymin, detection.xmax, detection.ymax])
                if self.source == Previews.color.name and not self._full_fov:
                    bbox[::2] += self._cropOffsetX(frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self._bbox_colors[detection.label], 2)
                cv2.rectangle(frame, (bbox[0], (bbox[1] - 28)), ((bbox[0] + 110), bbox[1]), self._bbox_colors[detection.label], cv2.FILLED)
                cv2.putText(frame, self.get_label_text(detection.label), (bbox[0] + 5, bbox[1] - 10),
                            self._text_type, 0.5, (0, 0, 0), 1, self._line_type)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 62, bbox[1] - 10),
                            self._text_type, 0.5, (0, 0, 0), 1, self._line_type)

                if hasattr(detection, 'spatialCoordinates'):  # Display spatial coordinates as well
                    x_meters = detection.spatialCoordinates.x / 1000
                    y_meters = detection.spatialCoordinates.y / 1000
                    z_meters = detection.spatialCoordinates.z / 1000
                    cv2.putText(frame, "X: {:.2f} m".format(x_meters), (bbox[0] + 10, bbox[1] + 60),
                                self._text_type, 0.5, self._text_bg_color, 4, self._line_type)
                    cv2.putText(frame, "X: {:.2f} m".format(x_meters), (bbox[0] + 10, bbox[1] + 60),
                                self._text_type, 0.5, self._text_color, 1, self._line_type)
                    cv2.putText(frame, "Y: {:.2f} m".format(y_meters), (bbox[0] + 10, bbox[1] + 75),
                                self._text_type, 0.5, self._text_bg_color, 4, self._line_type)
                    cv2.putText(frame, "Y: {:.2f} m".format(y_meters), (bbox[0] + 10, bbox[1] + 75),
                                self._text_type, 0.5, self._text_color, 1, self._line_type)
                    cv2.putText(frame, "Z: {:.2f} m".format(z_meters), (bbox[0] + 10, bbox[1] + 90),
                                self._text_type, 0.5, self._text_bg_color, 4, self._line_type)
                    cv2.putText(frame, "Z: {:.2f} m".format(z_meters), (bbox[0] + 10, bbox[1] + 90),
                                self._text_type, 0.5, self._text_color, 1, self._line_type)
            for detection in decoded_data:
                if isinstance(source, PreviewManager):
                    for name, frame in source.frames.items():
                        draw_detection(frame, detection)
                else:
                    draw_detection(source, detection)

            if self._count_label is not None:
                self._draw_count(source, decoded_data)

        elif self._output_format == "raw" and self._handler is not None:
            if isinstance(source, PreviewManager):
                frames = list(source.frames.items())
            else:
                frames = [("host", source)]
            self._handler.draw(self, decoded_data, frames)

    def createQueues(self, device):
        """
        Creates output queue for NeuralNetwork node and, if using :code:`host` as a :attr:`source`, it will also create
        input queue.

        Args:
            device (depthai.Device): Running device instance
        """
        if self.source == "host":
            self.input_queue = device.getInputQueue("nn_in", maxSize=1, blocking=False)
        self.output_queue = device.getOutputQueue("nn_out", maxSize=1, blocking=False)

    def sendInputFrame(self, frame, seq_num=None):
        """
        Sends a frame into :attr:`input_queue` object. Handles scaling down the frame, creating a proper :obj:`depthai.ImgFrame`
        and sending it to the queue. Be sure to use :code:`host` as a :attr:`source` and call :func:`createQueues` prior
        input queue.

        Args:
            frame (numpy.ndarray): Frame to be sent to the device
            seq_num (int, optional): Sequence number set on ImgFrame. Useful in syncronization scenarios

        Returns:
            numpy.ndarray: scaled frame that was sent to the NN (same width/height as NN input)

        Raises:
            RuntimeError: if :attr:`input_queue` is :code:`None` (unable to send the image)
        """
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

    def countLabel(self, label):
        """
        Enables object count for specific label. Label count will be printed once :func:`draw` method is called

        Args:
            label (str | int): Label to be counted. If model is using mappings in model config file, supply here a :obj:`str` label
                to be tracked. If no mapping is present, specify the label as :obj:`int` (NN-default)
        """

        self._count_label = label
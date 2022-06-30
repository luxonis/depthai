import json
from pathlib import Path
import depthai as dai
import cv2
import numpy as np

from .preview_manager import PreviewManager, SyncedPreviewManager
from ..previews import Previews
from ..utils import loadModule, toTensorResult, frameNorm, toPlanar


class NNetManager:
    """
    Manager class handling all NN-related functionalities. It's capable of creating appropriate nodes and connections,
    decoding neural network output automatically or by using external handler file.
    """

    def __init__(self, inputSize, nnFamily=None, labels=[], confidence=0.5, sync=False):
        """
        Args:
            inputSize (tuple): Desired NN input size, should match the input size defined in the network itself (width, height)
            nnFamily (str, Optional): type of NeuralNetwork to be processed. Supported: :code:`"YOLO"` and :code:`mobilenet`
            labels (list, Optional): Allows to display class label instead of ID when drawing nn detections.
            confidence (float, Optional): Specify detection nn's confidence threshold
            sync (bool, Optional): Store NN results for preview syncing (to be used with SyncedPreviewManager
        """
        self.inputSize = inputSize
        self._nnFamily = nnFamily
        if nnFamily in ("YOLO", "mobilenet"):
            self._outputFormat = "detection"
        self._labels = labels
        self._confidence = confidence
        self._sync = sync

    #: list: List of available neural network inputs
    sourceChoices = ("color", "left", "right", "rectifiedLeft", "rectifiedRight", "host")
    #: str: Selected neural network input
    source = None
    #: tuple: NN input size (width, height)
    inputSize = None
    #: depthai.OpenVINO.Version: OpenVINO version, available only if parsed from config file (see :func:`readConfig`)
    openvinoVersion = None
    #: depthai.DataInputQueue: DepthAI input queue object that allows to send images from host to device (used only with :code:`host` source)
    inputQueue = None
    #: depthai.DataOutputQueue: DepthAI output queue object that allows to receive NN results from the device.
    outputQueue = None
    #: dict: nn data buffer, disabled by default. Stores parsed nn data with packet sequence number as dict key
    buffer = {}


    _bboxColors = np.random.random(size=(256, 3)) * 256  # Random Colors for bounding boxes
    _countLabel = None
    _textBgColor = (0, 0, 0)
    _textColor = (255, 255, 255)
    _lineType = cv2.LINE_AA
    _textType = cv2.FONT_HERSHEY_SIMPLEX
    _outputFormat = "raw"
    _metadata = None
    _fullFov = True
    _config = None
    _nnFamily = None
    _handler = None

    def readConfig(self, path):
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
        configPath = Path(path)
        if not configPath.exists():
            raise ValueError("Path {} does not exist!".format(path))

        with configPath.open() as f:
            self._config = json.load(f)
            if "openvino_version" in self._config:
                self.openvinoVersion =getattr(dai.OpenVINO.Version, 'VERSION_' + self._config.get("openvino_version"))
            nnConfig = self._config.get("nn_config", {})
            self._labels = self._config.get("mappings", {}).get("labels", None)
            self._nnFamily = nnConfig.get("NN_family", None)
            self._outputFormat = nnConfig.get("output_format", "raw")
            self._metadata = nnConfig.get("NN_specific_metadata", {})
            if "input_size" in nnConfig:
                self.inputSize = tuple(map(int, nnConfig.get("input_size").split('x')))

            self._confidence = self._metadata.get("confidence_threshold", nnConfig.get("confidence_threshold", None))
            if 'handler' in self._config:
                self._handler = loadModule(configPath.parent / self._config["handler"])

                if not callable(getattr(self._handler, "draw", None)) or not callable(getattr(self._handler, "decode", None)):
                    raise RuntimeError("Custom model handler does not contain 'draw' or 'decode' methods!")


    def _normFrame(self, frame):
        if not self._fullFov:
            scaleF = frame.shape[0] / self.inputSize[1]
            return np.zeros((int(self.inputSize[1] * scaleF), int(self.inputSize[0] * scaleF)))
        else:
            return frame

    def _cropOffsetX(self, frame):
        if not self._fullFov:
            croppedW = (frame.shape[0] / self.inputSize[1]) * self.inputSize[0]
            return int((frame.shape[1] - croppedW) // 2)
        else:
            return 0

    def createNN(self, pipeline, nodes, blobPath, source="color", useDepth=False, minDepth=100, maxDepth=10000, sbbScaleFactor=0.3, fullFov=True, useImageManip=True):
        """
        Creates nodes and connections in provided pipeline that will allow to run NN model and consume it's results.

        Args:
            pipeline (depthai.Pipeline): Pipeline instance
            nodes (types.SimpleNamespace): Object cointaining all of the nodes added to the pipeline. Available in :attr:`depthai_sdk.managers.PipelineManager.nodes`
            blobPath (pathlib.Path): Path to MyriadX blob. Might be useful to use together with
                :func:`depthai_sdk.managers.BlobManager.getBlob()` for dynamic blob compilation
            source (str, Optional): Neural network input source, one of :attr:`sourceChoices`
            useDepth (bool, Optional): If set to True, produced detections will have spatial coordinates included
            minDepth (int, Optional): Minimum depth distance in centimeters
            maxDepth (int, Optional): Maximum depth distance in centimeters
            sbbScaleFactor (float, Optional): Scale of the bounding box that will be used to calculate spatial coordinates for
                detection. If set to 0.3, it will scale down center-wise the bounding box to 0.3 of it's original size
                and use it to calculate spatial location of the object
            fullFov (bool, Optional): If set to False, manager will include crop offset when scaling the detections.
                Usually should be set to True (if you don't perform aspect ratio crop or when `keepAspectRatio` flag
                on camera/manip node is set to False
            useImageManip (bool, Optional): If set to False, manager will not create an image manip node for input image
                scaling - which may result in an input image being not adjusted for the NeuralNetwork node. Can be useful
                when we want to limit the amount of nodes running simultaneously on device

        Returns:
            depthai.node.NeuralNetwork: Configured NN node that was added to the pipeline

        Raises:
            RuntimeError: If source is not a valid choice or when input size has not been set.
        """
        if source not in self.sourceChoices:
            raise RuntimeError(f"Source {source} is invalid, available {self.sourceChoices}")
        if self.inputSize is None:
            raise RuntimeError("Unable to determine the nn input size. Please use --cnnInputSize flag to specify it in WxH format: -nnSize <width>x<height>")

        self.source = source
        self._fullFov = fullFov
        if self._nnFamily == "mobilenet":
            nodes.nn = pipeline.createMobileNetSpatialDetectionNetwork() if useDepth else pipeline.createMobileNetDetectionNetwork()
            nodes.nn.setConfidenceThreshold(self._confidence)
        elif self._nnFamily == "YOLO":
            nodes.nn = pipeline.createYoloSpatialDetectionNetwork() if useDepth else pipeline.createYoloDetectionNetwork()
            nodes.nn.setConfidenceThreshold(self._confidence)
            nodes.nn.setNumClasses(self._metadata["classes"])
            nodes.nn.setCoordinateSize(self._metadata["coordinates"])
            nodes.nn.setAnchors(self._metadata["anchors"])
            nodes.nn.setAnchorMasks(self._metadata["anchor_masks"])
            nodes.nn.setIouThreshold(self._metadata["iou_threshold"])
        else:
            # TODO use createSpatialLocationCalculator
            nodes.nn = pipeline.createNeuralNetwork()

        nodes.nn.setBlobPath(str(blobPath))
        nodes.nn.setNumInferenceThreads(2)
        nodes.nn.input.setBlocking(False)
        nodes.nn.input.setQueueSize(2)

        nodes.xoutNn = pipeline.createXLinkOut()
        nodes.xoutNn.setStreamName("nnOut")
        nodes.nn.out.link(nodes.xoutNn.input)

        if self.source == "host":
            nodes.xinNn = pipeline.createXLinkIn()
            nodes.xinNn.setMaxDataSize(self.inputSize[0] * self.inputSize[1] * 3)
            nodes.xinNn.setStreamName("nnIn")
            nodes.xinNn.out.link(nodes.nn.input)
        else:
            if useImageManip:
                nodes.manipNn = pipeline.createImageManip()
                nodes.manipNn.initialConfig.setResize(*self.inputSize)
                # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
                nodes.manipNn.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
                # NN inputs
                nodes.manipNn.out.link(nodes.nn.input)
                nodes.manipNn.setKeepAspectRatio(not self._fullFov)
                nodes.manipNn.setMaxOutputFrameSize(self.inputSize[0] * self.inputSize[1] * 3)

                link_input = nodes.manipNn.inputImage
            else:
                link_input = nodes.nn.input

            if self.source == "color":
                nodes.camRgb.preview.link(link_input)
            if self.source == "left":
                nodes.monoLeft.out.link(link_input)
            elif self.source == "right":
                nodes.monoRight.out.link(link_input)
            elif self.source == "rectifiedLeft":
                nodes.stereo.rectifiedLeft.link(link_input)
            elif self.source == "rectifiedRight":
                nodes.stereo.rectifiedRight.link(link_input)

        if self._nnFamily in ("YOLO", "mobilenet") and useDepth:
            nodes.stereo.depth.link(nodes.nn.inputDepth)
            nodes.nn.setDepthLowerThreshold(minDepth)
            nodes.nn.setDepthUpperThreshold(maxDepth)
            nodes.nn.setBoundingBoxScaleFactor(sbbScaleFactor)

        return nodes.nn

    def getLabelText(self, label):
        """
        Retrieves text assigned to specific label

        Args:
            label (int): Integer representing detection label, usually returned from NN node

        Returns:
            str: Label text assigned to specific label id or label id

        Raises:
            RuntimeError: If source is not a valid choice or when input size has not been set.
        """
        if self._labels is None:
            return str(label)
        elif int(label) < len(self._labels):
            return self._labels[int(label)]
        else:
            print(f"Label of ouf bounds (label index: {label}, available labels: {len(self._labels)}")
            return str(label)

    def parse(self, blocking=False):
        if self.outputQueue is None:
            return None, None

        if blocking:
            inNn = self.outputQueue.get()
        else:
            inNn = self.outputQueue.tryGet()
        if inNn is not None:
            data = self.decode(inNn)
            if self._sync:
                self.buffer[inNn.getSequenceNum()] = data
            return data, inNn
        else:
            return None, None

    def decode(self, inNn):
        """
        Decodes NN output. Performs generic handling for supported detection networks or calls custom handler methods

        Args:
            inNn (depthai.NNData): Integer representing detection label, usually returned from NN node

        Returns:
            Decoded NN data

        Raises:
            RuntimeError: if outputFormat specified in model config file is not recognized
        """
        if self._outputFormat == "detection":
            return inNn.detections
        elif self._outputFormat == "raw":
            if self._handler is not None:
                return self._handler.decode(self, inNn)
            else:
                try:
                    data = toTensorResult(inNn)
                    print("Received NN packet: ", ", ".join([f"{key}: {value.shape}" for key, value in data.items()]))
                except Exception as ex:
                    print("Received NN packet: <Preview unabailable: {}>".format(ex))
        else:
            raise RuntimeError("Unknown output format: {}".format(self._outputFormat))

    def _drawCount(self, source, decodedData):
        def drawCnt(frame, cnt):
            cv2.putText(frame, f"{self._countLabel}: {cnt}", (5, 46), self._textType, 0.5, self._textBgColor, 4, self._lineType)
            cv2.putText(frame, f"{self._countLabel}: {cnt}", (5, 46), self._textType, 0.5, self._textColor, 1, self._lineType)

        # Count the number of detected objects
        cntList = list(filter(lambda x: self.getLabelText(x.label) == self._countLabel, decodedData))
        if isinstance(source, PreviewManager):
            for frame in source.frames.values():
                drawCnt(frame, len(cntList))
        else:
            drawCnt(source, len(cntList))

    def draw(self, source, decodedData):
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

            decodedData: Detections from neural network node, usually returned from :func:`decode` method
        """
        if self._outputFormat == "detection":
            def drawDetection(frame, detection):
                bbox = frameNorm(self._normFrame(frame), [detection.xmin, detection.ymin, detection.xmax, detection.ymax])
                if self.source == Previews.color.name and not self._fullFov:
                    bbox[::2] += self._cropOffsetX(frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self._bboxColors[detection.label], 2)
                cv2.rectangle(frame, (bbox[0], (bbox[1] - 28)), ((bbox[0] + 110), bbox[1]), self._bboxColors[detection.label], cv2.FILLED)
                cv2.putText(frame, self.getLabelText(detection.label), (bbox[0] + 5, bbox[1] - 10),
                            self._textType, 0.5, (0, 0, 0), 1, self._lineType)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 62, bbox[1] - 10),
                            self._textType, 0.5, (0, 0, 0), 1, self._lineType)

                if hasattr(detection, 'spatialCoordinates'):  # Display spatial coordinates as well
                    xMeters = detection.spatialCoordinates.x / 1000
                    yMeters = detection.spatialCoordinates.y / 1000
                    zMeters = detection.spatialCoordinates.z / 1000
                    cv2.putText(frame, "X: {:.2f} m".format(xMeters), (bbox[0] + 10, bbox[1] + 60),
                                self._textType, 0.5, self._textBgColor, 4, self._lineType)
                    cv2.putText(frame, "X: {:.2f} m".format(xMeters), (bbox[0] + 10, bbox[1] + 60),
                                self._textType, 0.5, self._textColor, 1, self._lineType)
                    cv2.putText(frame, "Y: {:.2f} m".format(yMeters), (bbox[0] + 10, bbox[1] + 75),
                                self._textType, 0.5, self._textBgColor, 4, self._lineType)
                    cv2.putText(frame, "Y: {:.2f} m".format(yMeters), (bbox[0] + 10, bbox[1] + 75),
                                self._textType, 0.5, self._textColor, 1, self._lineType)
                    cv2.putText(frame, "Z: {:.2f} m".format(zMeters), (bbox[0] + 10, bbox[1] + 90),
                                self._textType, 0.5, self._textBgColor, 4, self._lineType)
                    cv2.putText(frame, "Z: {:.2f} m".format(zMeters), (bbox[0] + 10, bbox[1] + 90),
                                self._textType, 0.5, self._textColor, 1, self._lineType)
            if isinstance(source, SyncedPreviewManager):
                if len(self.buffer) > 0 and source.nnSyncSeq is not None:
                    data = self.buffer.get(source.nnSyncSeq, self.buffer[max(self.buffer.keys())])
                    for old_key in list(filter(lambda key: key < source.nnSyncSeq, self.buffer.keys())):
                        del self.buffer[old_key]
                    for detection in data:
                        for name, frame in source.frames.items():
                            drawDetection(frame, detection)
            else:
                for detection in decodedData:
                    if isinstance(source, PreviewManager):
                        for name, frame in source.frames.items():
                            drawDetection(frame, detection)
                    else:
                        drawDetection(source, detection)

            if self._countLabel is not None:
                self._drawCount(source, decodedData)

        elif self._outputFormat == "raw" and self._handler is not None:
            if isinstance(source, PreviewManager):
                frames = list(source.frames.items())
            else:
                frames = [("host", source)]
            self._handler.draw(self, decodedData, frames)

    def createQueues(self, device):
        """
        Creates output queue for NeuralNetwork node and, if using :code:`host` as a :attr:`source`, it will also create
        input queue.

        Args:
            device (depthai.Device): Running device instance
        """
        if self.source == "host":
            self.inputQueue = device.getInputQueue("nnIn", maxSize=1, blocking=False)
        self.outputQueue = device.getOutputQueue("nnOut", maxSize=1, blocking=False)

    def closeQueues(self):
        """
        Closes output queues created by :func:`createQueues`
        """
        if self.source == "host" and self.inputQueue is not None:
            self.inputQueue.close()
        if self.outputQueue is not None:
            self.outputQueue.close()

    def sendInputFrame(self, frame, seqNum=None):
        """
        Sends a frame into :attr:`inputQueue` object. Handles scaling down the frame, creating a proper :obj:`depthai.ImgFrame`
        and sending it to the queue. Be sure to use :code:`host` as a :attr:`source` and call :func:`createQueues` prior
        input queue.

        Args:
            frame (numpy.ndarray): Frame to be sent to the device
            seqNum (int, Optional): Sequence number set on ImgFrame. Useful in synchronization scenarios

        Returns:
            numpy.ndarray: scaled frame that was sent to the NN (same width/height as NN input)

        Raises:
            RuntimeError: if :attr:`inputQueue` is :code:`None` (unable to send the image)
        """
        if self.inputQueue is None:
            raise RuntimeError("Unable to send image, no input queue is present! Call `createQueues(device)` first!")

        scaledFrame = cv2.resize(frame, self.inputSize)
        frameNn = dai.ImgFrame()
        if seqNum is not None:
            frameNn.setSequenceNum(seqNum)
        frameNn.setType(dai.ImgFrame.Type.BGR888p)
        frameNn.setWidth(self.inputSize[0])
        frameNn.setHeight(self.inputSize[1])
        frameNn.setData(toPlanar(scaledFrame))
        self.inputQueue.send(frameNn)

        return scaledFrame

    def countLabel(self, label):
        """
        Enables object count for specific label. Label count will be printed once :func:`draw` method is called

        Args:
            label (str | int): Label to be counted. If model is using mappings in model config file, supply here a :obj:`str` label
                to be tracked. If no mapping is present, specify the label as :obj:`int` (NN-default)
        """

        self._countLabel = label
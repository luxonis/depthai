import re
from enum import IntEnum
from .component import Component
from .camera_component import CameraComponent
from .stereo_component import StereoComponent
from .multi_stage_nn import MultiStageNN
from pathlib import Path
from typing import Callable, Optional, Union, List, Dict, Tuple
import depthai as dai
import os
import blobconverter
from ..utils import loadModule, isUrl, getBlob
from .parser import *
from ..classes.nn_config import Config, Model
import json


class AspectRatioResizeMode(IntEnum):
    """
    If NN input frame is in different aspect ratio than what the model expect, we have 3 different
    modes of operation. Full documentation:
    https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/
    """
    LETTERBOX = 0  # Preserves full FOV, but smaller frame means less features which might decrease NN accuracy
    STRETCH = 1  # Preserves full FOV, but frames are stretched which might decrease NN accuracy
    CROP = 2  # Crops some FOV to match the required FOV. No potential NN accuracy decrease.


class NNComponent(Component):
    tracker: dai.node.ObjectTracker
    node: Union[
        None,
        dai.node.NeuralNetwork,
        dai.node.MobileNetDetectionNetwork,
        dai.node.MobileNetSpatialDetectionNetwork,
        dai.node.YoloDetectionNetwork,
        dai.node.YoloSpatialDetectionNetwork,
    ] = None
    arResizeMode: AspectRatioResizeMode = AspectRatioResizeMode.LETTERBOX  # Default
    manip: dai.node.ImageManip = None  # ImageManip used to resize the input to match the expected NN input size

    # Setting passed at init or persed from these settings
    inputComponent: Optional[Component] = None  # Used for visualizer. Only set if component was passed as an input
    blob: dai.OpenVINO.Blob
    _forcedVersion: Optional[dai.OpenVINO.Version] = None  # Forced OpenVINO version
    _input: dai.Node.Output  # Original high-res input
    size: Tuple[int, int]  # Input size to the NN
    _args: Dict = None
    config: Config = None
    _xout: Union[None, bool, str] = None  # Argument passed by user
    _tracker: bool = False
    _spatial: Union[None, bool, StereoComponent, dai.Node.Output] = None
    _nodeType: dai.node = None  # Type of the node for `node`

    passthroughOut: bool = False  # Whether to stream passthrough frame to the host

    out: dai.Node.Output  # NN output
    passthrough: dai.Node.Output
    _multiStageNn: MultiStageNN

    # For visualizer
    labels: List = None  # obj detector labels
    handler: Callable = None  # Custom model handler for decoding

    def __init__(self,
                 model: Union[str, Path],  # str for SDK supported model or Path to custom model's json
                 input: Union[dai.Node.Output, Component],
                 out: Union[None, bool, str] = None,
                 nnType: Optional[str] = None,
                 tracker: bool = False,  # Enable object tracker - only for Object detection models
                 spatial: Union[None, bool, StereoComponent, dai.Node.Output] = None,
                 args: Dict = None  # User defined args
                 ) -> None:
        """
        Neural Network component that abstracts the following API nodes: NeuralNetwork, MobileNetDetectionNetwork,
        MobileNetSpatialDetectionNetwork, YoloDetectionNetwork, YoloSpatialDetectionNetwork, ObjectTracker
        (only for object detectors).

        Args:
            model (Union[str, Path]): str for SDK supported model / Path to blob or custom model's json
            input: (Union[Component, dai.Node.Output]): Input to the NN. If nn_component that is object detector, crop HQ frame at detections (Script node + ImageManip node)
            out (bool, default False): Stream component's output to the host
            nnType (str, optional): Type of the NN - Either Yolo or MobileNet
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            args (Any, optional): Set the camera components based on user arguments
        """
        super().__init__()

        # Save passed settings
        self.input = input
        self._spatial = spatial
        self._nnType = nnType
        self._xout = out
        self._args = args
        self._tracker = tracker

        # Parse passed settings
        self._blob = self._parseModel(model)

    def forcedOpenVinoVersion(self) -> dai.OpenVINO.Version:
        """
        Checks whether the component forces a specific OpenVINO version. This function is called after
        Camera has been configured and right before we connect to the OAK camera.
        @return: Forced OpenVINO version (optional).
        """
        return self._forcedVersion

    def updateDeviceInfo(self, pipeline: dai.Pipeline, device: dai.Device):
        # TODO: update NN input based on camera resolution
        if self.config and not self._nnType:
            nnConfig = self.config.get("nn_config", {})
            if 'NN_family' in nnConfig:
                self._nnType = str(nnConfig['NN_family']).upper()

        self.node = pipeline.create(self._nodeType)
        self.node.setBlob(self.blob)
        self.out = self.node.out

        if self.config:
            nnConfig = self.config.get("nn_config", {})
            if self.isDetector() and 'confidence_threshold' in nnConfig:
                self.node.setConfidenceThreshold(float(nnConfig['confidence_threshold']))

            meta = nnConfig.get('NN_specific_metadata', None)
            if self.isYolo() and meta:
                self.configYoloFromMeta(metadata=meta)

        if not self.node or not self.blob:
            raise Exception('Blob/Node not found!')

        if 1 < len(self.blob.networkInputs):
            raise NotImplementedError()

        nnIn: dai.TensorInfo = next(iter(self.blob.networkInputs.values()))
        # TODO: support models that expect mono img
        self.size: Tuple[int, int] = (nnIn.dims[0], nnIn.dims[1])
        # maxSize = dims

        if isinstance(self.input, CameraComponent):
            self.inputComponent = self.input
            self.input = self.input.out
            self._setupResizeManip(pipeline).link(self.node.input)
        elif isinstance(self.input, type(self)):
            if not self.input.isDetector():
                raise Exception('Only object detector models can be used as an input to the NNComponent!')
            self.inputComponent = self.input  # Used by visualizer
            # Create script node, get HQ frames from input.
            self._multiStageNn = MultiStageNN(pipeline, self.input, self.input.input, self.size)
            self._multiStageNn.out.link(self.node.input)  # Cropped frames
            # For debugging, for intenral counter
            self.node.out.link(self._multiStageNn.script.inputs['recognition'])
            self.node.input.setBlocking(True)
            self.node.input.setQueueSize(15)
        elif isinstance(self.input, dai.Node.Output):
            # Link directly via ImageManip
            self.input = self.input
            self._setupResizeManip(pipeline, self.size, self.input).link(self.node.input)

        if self._spatial:
            if isinstance(self._spatial, bool):  # Create new StereoComponent
                spatial = StereoComponent(pipeline, args=self._args)
            if isinstance(spatial, StereoComponent):
                spatial.depth.link(self.node.inputDepth)
            elif isinstance(spatial, dai.Node.Output):
                spatial.link(self.node.inputDepth)

        if self._spatial:
            if not self.isDetector():
                print('Currently, only object detector models (Yolo/MobileNet) can use tracker!')
            else:
                raise NotImplementedError()
                # self.tracker = pipeline.createObjectTracker()
                # self.out = self.tracker.out

        if self._xout:
            super().createXOut(
                pipeline,
                type(self),
                name=self._xout,
                out=self.out,
                depthaiMsg=dai.ImgDetections if self.isDetector() else dai.NNData
            )

        if self.passthroughOut:
            super().createXOut(
                pipeline,
                type(self),
                name='passthrough',
                out=self.node.passthrough,
                depthaiMsg=dai.ImgFrame
            )

    def _parseModel(self, model):
        """
        Called when NNComponent is initialized. Parses "model" argument passed by user.
        """
        # Parse the input config/model
        if isinstance(model, str):
            # Download from the web, or convert to Path
            model = getBlob(model) if isUrl(model) else Path(model)

        if model.is_file():
            if model.suffix == '.blob':
                self.blob = dai.OpenVINO.Blob(model)
                # BlobConverter sets name of the blob '[name]_openvino_[version]_[num]cores.blob'
                # So we can parse this openvino version if it exists
                match = re.search('_openvino_\d{4}.\d', str(model))
                if match is not None:
                    version = match.group().replace('_openvino_', '')
                    # Will force specific OpenVINO version
                    self._forcedVersion = parseOpenVinoVersion(version)
            elif model.suffix == '.json':  # json config file was passed
                self.parseConfig(model)
        else:  # SDK supported model
            availableModels = self.getSupportedModels(printModels=False)
            if str(model) not in availableModels:
                raise ValueError(f"Specified model '{str(model)}' is not supported by DepthAI SDK. \
                    Check SDK documentation page to see which models are supported.")

            model = availableModels[str(model)] / 'config.json'
            self.parseConfig(model)

    def setAspectRatioResizeMode(self, mode: AspectRatioResizeMode):
        self.arResizeMode = mode

    def parseNodeType(self, nnType: str) -> None:
        self._nodeType = dai.node.NeuralNetwork
        if nnType:
            if nnType.upper() == 'YOLO':
                self._nodeType = dai.node.YoloSpatialDetectionNetwork if self._spatial else dai.node.YoloDetectionNetwork
            elif nnType.upper() == 'MOBILENET':
                self._nodeType = dai.node.MobileNetSpatialDetectionNetwork if self._spatial else dai.node.MobileNetDetectionNetwork
    def parseConfig(self, modelConfig: Path):
        """
        Called when NNComponent is initialized. Reads config.json file and parses relevant setting from there
        """
        with modelConfig.open() as f:
            self.config = Config().load(json.loads(f.read()))

        # Get blob from the config file
        blob = self._blobFromConfig(self.config['model'], modelConfig.parent)
        self.blob = dai.OpenVINO.Blob(blob)

        # Parse OpenVINO version
        if "openvino_version" in self.config:
            self._forcedVersion = parseOpenVinoVersion(self.conf.get("openvino_version"))

        # Save for visualization
        self.labels = self.config.get("mappings", {}).get("labels", None)

        # Handler.py logic to decode raw NN results into standardized AI results
        if 'handler' in self.config:
            self.handler = loadModule(modelConfig.parent / self.config["handler"])

            if not callable(getattr(self.handler, "decode", None)):
                raise RuntimeError("Custom model handler does not contain 'decode' method!")

        # Parse node type
        nnFamily = self.config.get("nn_config", {}).get("NN_family", None)
        if nnFamily:
            self.parseNodeType(nnFamily)


    def _blobFromConfig(self, model: Dict, parent: Path) -> str:
        """
        Gets the blob from the config file.
        @param model:
        @param parent: Path to the parent folder where the json file is stored
        """
        if 'blob' in model:
            return str(parent / model['blob'])

        if 'model_name' in model:  # Use blobconverter to download the model
            zoo_type = model.get("zoo_type", 'intel')
            return blobconverter.from_zoo(model['model_name'],
                                          zoo_type=zoo_type,
                                          shaves=6  # TODO: Calulate ideal shave amount
                                          )

        if 'xml' in model and 'bin' in model:
            return blobconverter.from_zoo(model['model_name'],
                                          xml=model['xml'],
                                          bin=model['bin'],
                                          data_type="FP16",  # Myriad X
                                          shaves=6  # TODO: Calulate ideal shave amount
                                          )

        raise ValueError("Specified `model` values in json config files are incorrect!")

    def _setupResizeManip(self, pipeline: Optional[dai.Pipeline] = None) -> dai.Node.Output:
        """
        Creates ImageManip node that resizes the input to match the expected NN input size.
        DepthAI uses CHW (Planar) channel layout and BGR color order convention.
        """
        if not self.manip:
            self.manip = pipeline.create(dai.node.ImageManip)
            self.input.link(self.manip.inputImage)
            self.manip.setMaxOutputFrameSize(self.size[0] * self.size[1] * 3)
            self.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

        # Set Aspect Ratio resizing mode
        if self.arResizeMode == AspectRatioResizeMode.CROP:
            # Cropping is already the default mode of the ImageManip node
            self.manip.initialConfig.setResize(self.size)
        elif self.arResizeMode == AspectRatioResizeMode.LETTERBOX:
            self.manip.initialConfig.setResizeThumbnail(*self.size)
        elif self.arResizeMode == AspectRatioResizeMode.STRETCH:
            self.manip.initialConfig.setResize(self.size)
            self.manip.setKeepAspectRatio(False)  # Not keeping aspect ratio -> stretching the image

        return self.manip.out

    def configMultiStageCropping(self,
                                 debug=False,
                                 labels: Optional[List[int]] = None,
                                 scaleBb: Optional[Tuple[int, int]] = None,
                                 ) -> None:
        """
        For multi-stage NN pipelines. Available if the input to this NNComponent was another NN component.

        Args:
            debug (bool, default False): Debug script node
            labels (List[int], optional): Crop & run inference only on objects with these labels
            scaleBb (Tuple[int, int], optional): Scale detection bounding boxes (x, y) before cropping the frame. In %.
        """
        if not isinstance(self.input, type(self)):
            print(
                "Input to this model was not a NNComponent, so 2-stage NN inferencing isn't possible! This configuration attempt will be ignored.")
            return

        self._multiStageNn.configMultiStageNn(debug, labels, scaleBb)

    @staticmethod
    def getSupportedModels(printModels=True) -> Dict[str, Path]:
        folder = Path(os.path.dirname(__file__)).parent / "nn_models"
        d = dict()
        for item in folder.iterdir():
            if item.is_dir() and item.name != '__pycache__':
                d[item.name] = item

        if printModels:
            print("\nDepthAI SDK supported models:\n")
            [print(f"- {name}") for name in d]
            print('')

        return d

    def configTracker(self,
                      type: Optional[dai.TrackerType] = None,
                      trackLabels: Optional[List[int]] = None,
                      assignmentPolicy: Optional[dai.TrackerIdAssignmentPolicy] = None,
                      maxObj: Optional[int] = None,
                      threshold: Optional[float] = None
                      ):
        """
        Configure object tracker if it's enabled.

        Args:
            type (dai.TrackerType, optional): Set object tracker type
            trackLabels (List[int], optional): Set detection labels to track
            assignmentPolicy (dai.TrackerType, optional): Set object tracker ID assignment policy
            maxObj (int, optional): Set set max objects to track. Max 60.
            threshold (float, optional): Set threshold for object detection confidence. Default: 0.0
        """

        if self.tracker is None:
            print(
                "Tracker was not enabled! Enable with cam.create_nn('[model]', tracker=True). This configuration attempt will be ignored.")
            return

        if type:
            self.tracker.setTrackerType(type=type)
        if trackLabels:
            self.tracker.setDetectionLabelsToTrack(trackLabels)
        if assignmentPolicy:
            self.tracker.setTrackerIdAssignmentPolicy(assignmentPolicy)
        if maxObj:
            if 60 < maxObj:
                raise ValueError("Maximum objects to track is 60!")
            self.tracker.setMaxObjectsToTrack(maxObj)
        if threshold:
            self.tracker.setTrackerThreshold(threshold)

    def configYoloFromMeta(self, metadata: Dict):
        return self._configYolo(
            numClasses=metadata['classes'],
            coordinateSize=metadata['coordinates'],
            anchors=metadata['anchors'],
            masks=metadata['anchor_masks'],
            iouThreshold=metadata['iou_threshold'],
            confThreshold=metadata['confidence_threshold'],
        )

    def _configYolo(self,
                    numClasses: int,
                    coordinateSize: int,
                    anchors: List[float],
                    masks: Dict[str, List[int]],
                    iouThreshold: float,
                    confThreshold: Optional[float] = None,
                    ) -> None:
        if not self.isYolo():
            print('This is not a YOLO detection network! This configuration attempt will be ignored.')
            return

        self.node.setNumClasses(numClasses)
        self.node.setCoordinateSize(coordinateSize)
        self.node.setAnchors(anchors)
        self.node.setAnchorMasks(masks)
        self.node.setIouThreshold(iouThreshold)

        if confThreshold: self.node.setConfidenceThreshold(confThreshold)

    def configNn(self,
                 passthroughOut: bool = False
                 ):

        self.passthroughOut = passthroughOut

    def configSpatial(self,
                      bbScaleFactor: Optional[float] = None,
                      lowerThreshold: Optional[int] = None,
                      upperThreshold: Optional[int] = None,
                      calcAlgo: Optional[dai.SpatialLocationCalculatorAlgorithm] = None,
                      out: Optional[Tuple[str, str]] = None
                      ) -> None:
        """
        Configures the Spatial NN network.
        Args:
            bbScaleFactor (float, optional): Specifies scale factor for detected bounding boxes (0..1]
            lowerThreshold (int, optional): Specifies lower threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            upperThreshold (int, optional): Specifies upper threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            calcAlgo (dai.SpatialLocationCalculatorAlgorithm, optional): Specifies spatial location calculator algorithm: Average/Min/Max
            out (Tuple[str, str], optional): Enable streaming depth + bounding boxes mappings to the host. Useful for debugging.
        """
        if not self.isSpatial():
            print('This is not a Spatial Detection network! This configuration attempt will be ignored.')
            return

        if bbScaleFactor: self.node.setBoundingBoxScaleFactor(bbScaleFactor)
        if lowerThreshold: self.node.setDepthLowerThreshold(lowerThreshold)
        if upperThreshold: self.node.setDepthUpperThreshold(upperThreshold)
        if calcAlgo: self.node.setSpatialCalculationAlgorithm(calcAlgo)
        if out:
            super().createXOut(
                self.pipeline,
                type(self),
                name=True if isinstance(out, bool) else out[0],
                out=self.node.passthroughDepth,
                depthaiMsg=dai.ImgFrame
            )
            super().createXOut(
                self.pipeline,
                type(self),
                name=True if isinstance(out, bool) else out[1],
                out=self.node.boundingBoxMapping,
                depthaiMsg=dai.SpatialLocationCalculatorConfig
            )

    def isSpatial(self) -> bool:
        return (
                isinstance(self.node, dai.node.MobileNetSpatialDetectionNetwork) or \
                isinstance(self.node, dai.node.YoloSpatialDetectionNetwork)
        )

    def isYolo(self) -> bool:
        return (
                isinstance(self.node, dai.node.YoloDetectionNetwork) or \
                isinstance(self.node, dai.node.YoloSpatialDetectionNetwork)
        )

    def isMobileNet(self) -> bool:
        return (
                isinstance(self.node, dai.node.MobileNetDetectionNetwork) or \
                isinstance(self.node, dai.node.MobileNetSpatialDetectionNetwork)
        )

    def isDetector(self) -> bool:
        """
        Currently these 2 object detectors are supported
        """
        return self.isYolo() or self.isMobileNet()

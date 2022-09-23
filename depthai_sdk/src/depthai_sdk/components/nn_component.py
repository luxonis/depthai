import re
from .component import Component
from .camera_component import CameraComponent
from .stereo_component import StereoComponent
from .multi_stage_nn import MultiStageNN, MultiStageConfig
from pathlib import Path
from typing import Callable, Union, List, Dict

import blobconverter
from .parser import *
from .nn_helper import *
from ..classes.nn_config import Config
import json

from ..classes.xout import XoutNnResults, XoutTwoStage, XoutSpatialBbMappings
from ..classes.xout_base import StreamXout, XoutBase


class NNComponent(Component):
    # User accessible properties
    node: Union[
        None,
        dai.node.NeuralNetwork,
        dai.node.MobileNetDetectionNetwork,
        dai.node.MobileNetSpatialDetectionNetwork,
        dai.node.YoloDetectionNetwork,
        dai.node.YoloSpatialDetectionNetwork,
    ] = None
    tracker: dai.node.ObjectTracker = None
    arResizeMode: AspectRatioResizeMode = AspectRatioResizeMode.LETTERBOX  # Default
    manip: dai.node.ImageManip = None  # ImageManip used to resize the input to match the expected NN input size

    _input: Union[CameraComponent, 'NNComponent'] # Input to the NNComponent node passed on initialization
    _stream_input: dai.Node.Output # Node Output that will be used as the input for this NNComponent

    _blob: dai.OpenVINO.Blob = None
    _forcedVersion: Optional[dai.OpenVINO.Version] = None  # Forced OpenVINO version
    size: Tuple[int, int]  # Input size to the NN
    _args: Dict = None
    _config: Dict = None
    _tracker: bool = False
    _nodeType: dai.node = dai.node.NeuralNetwork  # Type of the node for `node`

    _multiStageNn: MultiStageNN = None
    _multi_stage_config: MultiStageConfig = None

    _spatial: Union[None, bool, StereoComponent] = None
    _spatial_config: SpatialConfig = None

    # For visualizer
    labels: List = None  # obj detector labels
    handler: Callable = None  # Custom model handler for decoding

    def __init__(self,
                 model: Union[str, Path, Dict],  # str for SDK supported model or Path to custom model's json
                 input: Union[CameraComponent, 'NNComponent'],
                 nnType: Optional[str] = None, # Either 'yolo' or 'mobilenet'
                 tracker: bool = False,  # Enable object tracker - only for Object detection models
                 spatial: Union[None, bool, StereoComponent] = None,
                 args: Dict = None  # User defined args
                 ) -> None:
        """
        Neural Network component that abstracts the following API nodes: NeuralNetwork, MobileNetDetectionNetwork,
        MobileNetSpatialDetectionNetwork, YoloDetectionNetwork, YoloSpatialDetectionNetwork, ObjectTracker
        (only for object detectors).

        Args:
            model (Union[str, Path, Dict]): str for SDK supported model / Path to blob or custom model's json
            input: (Union[Component, dai.Node.Output]): Input to the NN. If nn_component that is object detector, crop HQ frame at detections (Script node + ImageManip node)
            nnType (str, optional): Type of the NN - Either Yolo or MobileNet
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            args (Any, optional): Set the camera components based on user arguments
        """
        super().__init__()

        # Save passed settings
        self._input = input
        self._spatial = spatial
        self._args = args
        self._tracker = tracker

        # Parse passed settings
        self._parseModel(model)
        if nnType:
            self._parse_node_type(nnType)

    def _forced_openvino_version(self) -> dai.OpenVINO.Version:
        """
        Checks whether the component forces a specific OpenVINO version. This function is called after
        Camera has been configured and right before we connect to the OAK camera.
        @return: Forced OpenVINO version (optional).
        """
        return self._forcedVersion

    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):

        if self._blob is None:
            self._blob = dai.OpenVINO.Blob(self._blobFromConfig(self._config['model'], version))

        # TODO: update NN input based on camera resolution

        self.node = pipeline.create(self._nodeType)
        self.node.setBlob(self._blob)
        self._out = self.node.out

        if self._config:
            nnConfig = self._config.get("nn_config", {})
            if self.isDetector() and 'confidence_threshold' in nnConfig:
                self.node.setConfidenceThreshold(float(nnConfig['confidence_threshold']))

            meta = nnConfig.get('NN_specific_metadata', None)
            if self._isYolo() and meta:
                self.config_yolo_from_metadata(metadata=meta)

        if 1 < len(self._blob.networkInputs):
            raise NotImplementedError()

        nnIn: dai.TensorInfo = next(iter(self._blob.networkInputs.values()))
        # TODO: support models that expect mono img
        self.size: Tuple[int, int] = (nnIn.dims[0], nnIn.dims[1])
        # maxSize = dims

        if isinstance(self._input, CameraComponent):
            self._stream_input = self._input.out
            self._setupResizeManip(pipeline).link(self.node.input)

        elif self._isMultiStage():
            # Calculate crop shape of the object detector
            frameSize = self._input._input.out_size
            nnSize = self._input.size
            scale = frameSize[0] / nnSize[0], frameSize[1] / nnSize[1]
            i = 0 if scale[0] < scale[1] else 1
            crop = int(scale[i] * nnSize[0]), int(scale[i] * nnSize[1])
            # Crop the high-resolution frames so it matches object detection frame shape
            crop_manip = pipeline.createImageManip()
            crop_manip.setResize(*crop)
            crop_manip.setMaxOutputFrameSize(crop[0] * crop[1] * 3)
            crop_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            self._input._stream_input.link(crop_manip.inputImage)

            # Create script node, get HQ frames from input.
            self._multiStageNn = MultiStageNN(pipeline, self._input.node, crop_manip.out, self.size)
            self._multiStageNn.configure(self._multi_stage_config)
            self._multiStageNn.out.link(self.node.input)  # Cropped frames
            # For debugging, for intenral counter
            self.node.out.link(self._multiStageNn.script.inputs['recognition'])
            self.node.input.setBlocking(True)
            self.node.input.setQueueSize(15)

        if self._spatial:
            if isinstance(self._spatial, bool):  # Create new StereoComponent
                self._spatial = StereoComponent(args=self._args)
                self._spatial._update_device_info(pipeline, device, version)
            if isinstance(self._spatial, StereoComponent):
                self._spatial.depth.link(self.node.inputDepth)
                self._spatial.configure_stereo(align=self._input._source)
            # Configure Spatial Detection Network
            self._update_spatial()

        if self._tracker:
            if not self.isDetector():
                raise ValueError('Currently, only object detector models (Yolo/MobileNet) can use tracker!')
            raise NotImplementedError()
            # self.tracker = pipeline.createObjectTracker()
            # self.out = self.tracker.out

    def _parseModel(self, model):
        """
        Called when NNComponent is initialized. Parses "model" argument passed by user.
        """
        if isinstance(model, Dict):
            self.parse_config(model)
            return
        # Parse the input config/model
        elif isinstance(model, str):
            # Download from the web, or convert to Path
            model = getBlob(model) if isUrl(model) else Path(model)

        if model.suffix in ['.blob', '.json']:
            if model.suffix == '.blob':
                self._blob = dai.OpenVINO.Blob(model.resolve())
                # BlobConverter sets name of the blob '[name]_openvino_[version]_[num]cores.blob'
                # So we can parse this openvino version if it exists
                match = re.search('_openvino_\d{4}.\d', str(model))
                if match is not None:
                    version = match.group().replace('_openvino_', '')
                    # Will force specific OpenVINO version
                    self._forcedVersion = parseOpenVinoVersion(version)
            elif model.suffix == '.json':  # json config file was passed
                self.parse_config(model)
        else:  # SDK supported model
            models = getSupportedModels(printModels=False)
            if str(model) not in models:
                raise ValueError(f"Specified model '{str(model)}' is not supported by DepthAI SDK. \
                    Check SDK documentation page to see which models are supported.")

            model = models[str(model)] / 'config.json'
            self.parse_config(model)

    def _parse_node_type(self, nnType: str) -> None:
        self._nodeType = dai.node.NeuralNetwork
        if nnType:
            if nnType.upper() == 'YOLO':
                self._nodeType = dai.node.YoloSpatialDetectionNetwork if self._spatial else dai.node.YoloDetectionNetwork
            elif nnType.upper() == 'MOBILENET':
                self._nodeType = dai.node.MobileNetSpatialDetectionNetwork if self._spatial else dai.node.MobileNetDetectionNetwork

    def parse_config(self, modelConfig: Union[Path, str, Dict]):
        """
        Called when NNComponent is initialized. Reads config.json file and parses relevant setting from there
        """
        parentFolder = None
        if isinstance(modelConfig, str):
            modelConfig = Path(modelConfig).resolve()
        if isinstance(modelConfig, Path):
            parentFolder = modelConfig.parent
            with modelConfig.open() as f:
                self._config = Config().load(json.loads(f.read()))
        else:  # Dict
            self._config = modelConfig

        # Get blob from the config file
        if 'model' in self._config:
            model = self._config['model']

            # Resolve the paths inside config
            if parentFolder:
                for name in ['blob', 'xml', 'bin']:
                    if name in model:
                        model[name] = str((parentFolder / model[name]).resolve())

            if 'blob' in model:
                self._blob = dai.OpenVINO.Blob(model['blob'])

        # Parse OpenVINO version
        if "openvino_version" in self._config:
            self._forcedVersion = parseOpenVinoVersion(self.conf.get("openvino_version"))

        # Save for visualization
        self.labels = self._config.get("mappings", {}).get("labels", None)

        # Handler.py logic to decode raw NN results into standardized AI results
        if 'handler' in self._config:
            self.handler = loadModule(modelConfig.parent / self._config["handler"])

            if not callable(getattr(self.handler, "decode", None)):
                raise RuntimeError("Custom model handler does not contain 'decode' method!")

        # Parse node type
        nnFamily = self._config.get("nn_config", {}).get("NN_family", None)
        if nnFamily:
            self._parse_node_type(nnFamily)


    def _blobFromConfig(self, model: Dict, version: dai.OpenVINO.Version) -> str:
        """
        Gets the blob from the config file.
        @param model:
        @param parent: Path to the parent folder where the json file is stored
        """
        vals = str(version).split('_')
        versionStr = f"{vals[1]}.{vals[2]}"

        if 'model_name' in model:  # Use blobconverter to download the model
            zoo_type = model.get("zoo_type", 'intel')
            return blobconverter.from_zoo(model['model_name'],
                                          zoo_type=zoo_type,
                                          shaves=6,  # TODO: Calculate ideal shave amount
                                          version=versionStr
                                          )

        if 'xml' in model and 'bin' in model:
            return blobconverter.from_openvino(xml=model['xml'],
                                               bin=model['bin'],
                                               data_type="FP16",  # Myriad X
                                               shaves=6,  # TODO: Calculate ideal shave amount
                                               version=versionStr
                                               )

        raise ValueError("Specified `model` values in json config files are incorrect!")

    def _setupResizeManip(self, pipeline: Optional[dai.Pipeline] = None) -> dai.Node.Output:
        """
        Creates ImageManip node that resizes the input to match the expected NN input size.
        DepthAI uses CHW (Planar) channel layout and BGR color order convention.
        """
        if not self.manip:
            self.manip = pipeline.create(dai.node.ImageManip)
            self._stream_input.link(self.manip.inputImage)
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

    def config_multistage_nn(self,
                             debug=False,
                             show_cropped_frames=False,
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
        if not isinstance(self._input, type(self)):
            print("Input to this model was not a NNComponent, so 2-stage NN inferencing isn't possible! This configuration attempt will be ignored.")
            return

        self._multi_stage_config = MultiStageConfig(debug, show_cropped_frames, labels, scaleBb)

    def config_tracker(self,
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
            maxObj (int, optional): Set max objects to track. Max 60.
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

    def config_yolo_from_metadata(self, metadata: Dict):
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
        if not self._isYolo():
            print('This is not a YOLO detection network! This configuration attempt will be ignored.')
            return

        self.node.setNumClasses(numClasses)
        self.node.setCoordinateSize(coordinateSize)
        self.node.setAnchors(anchors)
        self.node.setAnchorMasks(masks)
        self.node.setIouThreshold(iouThreshold)

        if confThreshold: self.node.setConfidenceThreshold(confThreshold)

    def config_nn(self,
                  aspectRatioResizeMode: AspectRatioResizeMode = None,
                  ):

        if aspectRatioResizeMode:
            self.arResizeMode = aspectRatioResizeMode

    def config_spatial(self,
                       bbScaleFactor: Optional[float] = None,
                       lowerThreshold: Optional[int] = None,
                       upperThreshold: Optional[int] = None,
                       calcAlgo: Optional[dai.SpatialLocationCalculatorAlgorithm] = None,
                       out: bool = False
                       ):
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

        config = SpatialConfig()
        config.bbScaleFactor = bbScaleFactor
        config.lowerThreshold = lowerThreshold
        config.upperThreshold = upperThreshold
        config.calcAlgo = calcAlgo
        config.out = out
        self._spatial_config = config

    def _update_spatial(self):
        # Nothing to configure
        if self._spatial_config is None:
            return

        cfg = self._spatial_config
        if cfg.bbScaleFactor: self.node.setBoundingBoxScaleFactor(cfg.bbScaleFactor)
        if cfg.lowerThreshold: self.node.setDepthLowerThreshold(cfg.lowerThreshold)
        if cfg.upperThreshold: self.node.setDepthUpperThreshold(cfg.upperThreshold)
        if cfg.calcAlgo: self.node.setSpatialCalculationAlgorithm(cfg.calcAlgo)

    """
    Available outputs (to the host) of this component
    """
    def out(self, pipeline: dai.Pipeline, callback: Callable) -> XoutBase:
        # Check if it's XoutNnResults or XoutTwoStage

        if self._isMultiStage():
            out = XoutTwoStage(self._input, self, callback,
                               self._input._input.get_stream_xout(), # CameraComponent
                               StreamXout(self._input.node.id, self._input.node.out), # NnComponent (detections)
                               StreamXout(self.node.id, self.node.out), # This NnComponent (2nd stage NN)
                               )
        else:
            out = XoutNnResults(self, callback,
                                self._input.get_stream_xout(), # CameraComponent
                                StreamXout(self.node.id, self.node.out)) # NnComponent
        super()._create_xout(pipeline, out)
        return out

    def out_passthrough(self, pipeline: dai.Pipeline, callback: Callable) -> XoutNnResults:
        if self._isMultiStage():
            out = XoutTwoStage(self._input, self, callback,
                               StreamXout(self._input.node.id, self._input.node.passthrough), # Passthrough frame
                               StreamXout(self._input.node.id, self._input.node.out), # NnComponent (detections)
                               StreamXout(self.node.id, self.node.out), # This NnComponent (2nd stage NN)
                               )
        else:
            out = XoutNnResults(self, callback,
                                StreamXout(self.node.id, self.node.passthrough),
                                StreamXout(self.node.id, self.node.out))

        super()._create_xout(pipeline, out)
        return out

    def out_spatials(self, pipeline: dai.Pipeline, callback: Callable) -> XoutSpatialBbMappings:
        if not self.isSpatial():
            raise ValueError('SDK tried to output spatial data (depth + bounding box mappings), but this is not a Spatial Detection network!')

        out = XoutSpatialBbMappings(callback,
                                    StreamXout(self.node.id, self.node.passthroughDepth),
                                    StreamXout(self.node.id, self.node.boundingBoxMapping)
                                    )
        super()._create_xout(pipeline, out)
        return out

    def isSpatial(self) -> bool:
        return (
                self._nodeType == dai.node.MobileNetSpatialDetectionNetwork or
                self._nodeType == dai.node.YoloSpatialDetectionNetwork
        )

    def _isYolo(self) -> bool:
        return (
                self._nodeType == dai.node.YoloDetectionNetwork or
                self._nodeType == dai.node.YoloSpatialDetectionNetwork
        )

    def _isMobileNet(self) -> bool:
        return (
                self._nodeType == dai.node.MobileNetDetectionNetwork or
                self._nodeType == dai.node.MobileNetSpatialDetectionNetwork
        )

    def isDetector(self) -> bool:
        """
        Currently these 2 object detectors are supported
        """
        return self._isYolo() or self._isMobileNet()

    def _isMultiStage(self):
        if not isinstance(self._input, type(self)):
            return False

        if not self._input.isDetector():
            raise Exception('Only object detector models can be used as an input to the NNComponent!')

        return True

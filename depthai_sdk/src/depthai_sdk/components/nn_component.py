import json
from pathlib import Path
from typing import Callable, Union, List, Dict

import blobconverter

from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.components.multi_stage_nn import MultiStageNN, MultiStageConfig
from depthai_sdk.components.nn_helper import *
from depthai_sdk.components.parser import *
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.classes.nn_config import Config
from depthai_sdk.oak_outputs.xout import XoutNnResults, XoutTwoStage, XoutSpatialBbMappings, XoutFrames, XoutTracker
from depthai_sdk.oak_outputs.xout_base import StreamXout, XoutBase
from depthai_sdk.replay import Replay


class NNComponent(Component):
    # Public properties
    node: Union[
        None,
        dai.node.NeuralNetwork,
        dai.node.MobileNetDetectionNetwork,
        dai.node.MobileNetSpatialDetectionNetwork,
        dai.node.YoloDetectionNetwork,
        dai.node.YoloSpatialDetectionNetwork,
    ]
    tracker: dai.node.ObjectTracker
    imageManip: dai.node.ImageManip = None  # ImageManip used to resize the input to match the expected NN input size

    # Private properties
    _arResizeMode: AspectRatioResizeMode = AspectRatioResizeMode.LETTERBOX  # Default
    _input: Union[CameraComponent, 'NNComponent']  # Input to the NNComponent node passed on initialization
    _stream_input: dai.Node.Output  # Node Output that will be used as the input for this NNComponent

    _blob: dai.OpenVINO.Blob = None
    _forcedVersion: Optional[dai.OpenVINO.Version] = None  # Forced OpenVINO version
    _size: Tuple[int, int]  # Input size to the NN
    _args: Dict = None
    _config: Dict = None
    _nodeType: dai.node = dai.node.NeuralNetwork  # Type of the node for `node`

    _multiStageNn: MultiStageNN = None
    _multi_stage_config: MultiStageConfig = None

    _spatial: Union[None, bool, StereoComponent] = None
    _replay: Replay  # Replay module

    # For visualizer
    _labels: List = None  # obj detector labels
    _handler: Callable = None  # Custom model handler for decoding

    def __init__(self,
                 pipeline: dai.Pipeline,
                 model: Union[str, Path, Dict],  # str for SDK supported model or Path to custom model's json
                 input: Union[CameraComponent, 'NNComponent'],
                 nnType: Optional[str] = None,  # Either 'yolo' or 'mobilenet'
                 tracker: bool = False,  # Enable object tracker - only for Object detection models
                 spatial: Union[None, bool, StereoComponent] = None,
                 replay: Optional[Replay] = None,
                 args: Dict = None  # User defined args
                 ) -> None:
        """
        Neural Network component abstracts:
         - DepthAI API nodes: NeuralNetwork, *DetectionNetwork, *SpatialDetectionNetwork, ObjectTracker
         - Downloading NN models (supported SDK NNs), parsing NN json configs and setting up the pipeline based on it
         - Decoding NN results
         - MultiStage pipelines - cropping high-res frames based on detections and use them for second NN inferencing

        Args:
            model (Union[str, Path, Dict]): str for SDK supported model / Path to blob or custom model's json
            input: (Union[Component, dai.Node.Output]): Input to the NN. If nn_component that is object detector, crop HQ frame at detections (Script node + ImageManip node)
            nnType (str, optional): Type of the NN - Either 'Yolo' or 'MobileNet'
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            replay (Replay object): Replay
            args (Any, optional): Use user defined arguments when constructing the pipeline
        """
        super().__init__()
        self.out = self.Out(self)

        # Save passed settings
        self._input = input
        self._spatial = spatial
        self._args = args
        self._replay = replay

        self.tracker = pipeline.createObjectTracker() if tracker else None

        # Parse passed settings
        self._parse_model(model)
        if nnType:
            self._parse_node_type(nnType)

        # Create NN node
        self.node = pipeline.create(self._nodeType)

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
        self.node.setBlob(self._blob)
        self._out = self.node.out

        if self._config:
            nnConfig = self._config.get("nn_config", {})
            if self._isDetector() and 'confidence_threshold' in nnConfig:
                self.node.setConfidenceThreshold(float(nnConfig['confidence_threshold']))

            meta = nnConfig.get('NN_specific_metadata', None)
            if self._isYolo() and meta:
                self.config_yolo_from_metadata(metadata=meta)

        if 1 < len(self._blob.networkInputs):
            raise NotImplementedError()

        nnIn: dai.TensorInfo = next(iter(self._blob.networkInputs.values()))
        # TODO: support models that expect mono img
        self._size: Tuple[int, int] = (nnIn.dims[0], nnIn.dims[1])
        # maxSize = dims

        if isinstance(self._input, CameraComponent):
            self._stream_input = self._input._out
            self._setupResizeManip(pipeline).link(self.node.input)
        elif self._isMultiStage():
            # Calculate crop shape of the object detector
            frameSize = self._input._input._out_size
            nnSize = self._input._size
            scale = frameSize[0] / nnSize[0], frameSize[1] / nnSize[1]
            i = 0 if scale[0] < scale[1] else 1
            crop = int(scale[i] * nnSize[0]), int(scale[i] * nnSize[1])

            # Crop the high-resolution frames so it matches object detection frame aspect ratio
            self.imageManip = pipeline.createImageManip()
            self.imageManip.setResize(*crop)
            self.imageManip.setMaxOutputFrameSize(crop[0] * crop[1] * 3)
            self.imageManip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            self._input._stream_input.link(self.imageManip.inputImage)

            # Create script node, get HQ frames from input.
            self._multiStageNn = MultiStageNN(pipeline, self._input.node, self.imageManip.out, self._size)
            self._multiStageNn.configure(self._multi_stage_config)
            self._multiStageNn.out.link(self.node.input)  # Cropped frames
            # For debugging, for integral counter
            self.node.out.link(self._multiStageNn.script.inputs['recognition'])
            self.node.input.setBlocking(True)
            self.node.input.setQueueSize(15)
        else:
            raise ValueError(
                "'input' argument passed on init isn't supported! You can only use NnComponent or CameraComponent as the input.")

        if self._spatial:
            if isinstance(self._spatial, bool):  # Create new StereoComponent
                self._spatial = StereoComponent(pipeline, args=self._args, replay=self._replay)
                self._spatial._update_device_info(pipeline, device, version)
            if isinstance(self._spatial, StereoComponent):
                self._spatial.depth.link(self.node.inputDepth)
                self._spatial.config_stereo(align=self._input._source)
            # Configure Spatial Detection Network

        if self._args:
            if self._isSpatial():
                self._config_spatials_args(self._args)

    def _parse_model(self, model):
        """
        Called when NNComponent is initialized. Parses "model" argument passed by user.
        """
        if isinstance(model, Dict):
            self._parse_config(model)
            return
        # Parse the input config/model
        elif isinstance(model, str):
            # Download from the web, or convert to Path
            model = getBlob(model) if isUrl(model) else Path(model)

        if model.suffix in ['.blob', '.json']:
            if model.suffix == '.blob':
                self._blob = dai.OpenVINO.Blob(model.resolve())
                self._forcedVersion = self._blob.version
            elif model.suffix == '.json':  # json config file was passed
                self._parse_config(model)
        else:  # SDK supported model
            models = getSupportedModels(printModels=False)
            if str(model) not in models:
                raise ValueError(f"Specified model '{str(model)}' is not supported by DepthAI SDK. \
                    Check SDK documentation page to see which models are supported.")

            model = models[str(model)] / 'config.json'
            self._parse_config(model)

    def _parse_node_type(self, nnType: str) -> None:
        self._nodeType = dai.node.NeuralNetwork
        if nnType:
            if nnType.upper() == 'YOLO':
                self._nodeType = dai.node.YoloSpatialDetectionNetwork if self._isSpatial() else dai.node.YoloDetectionNetwork
            elif nnType.upper() == 'MOBILENET':
                self._nodeType = dai.node.MobileNetSpatialDetectionNetwork if self._isSpatial() else dai.node.MobileNetDetectionNetwork

    def _config_spatials_args(self, args):
        if not isinstance(args, Dict):
            args = vars(args)  # Namespace -> Dict
        self.config_spatial(
            bbScaleFactor=args.get('sbbScaleFactor', None),
            lowerThreshold=args.get('minDepth', None),
            upperThreshold=args.get('maxDepth', None),
        )

    def _parse_config(self, modelConfig: Union[Path, str, Dict]):
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
            self._forcedVersion = parseOpenVinoVersion(self._config.get("openvino_version"))

        # Save for visualization
        self._labels = self._config.get("mappings", {}).get("labels", None)

        # Handler.py logic to decode raw NN results into standardized AI results
        if 'handler' in self._config:
            self._handler = loadModule(modelConfig.parent / self._config["handler"])

            if not callable(getattr(self._handler, "decode", None)):
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
            zoo_type = model.get("zoo", 'intel')
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
        if not self.imageManip:
            self.imageManip = pipeline.create(dai.node.ImageManip)
            self._stream_input.link(self.imageManip.inputImage)
            self.imageManip.setMaxOutputFrameSize(self._size[0] * self._size[1] * 3)
            self.imageManip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            # Set to non-blocking
            self.imageManip.inputImage.setBlocking(False)
            self.imageManip.inputImage.setQueueSize(2)

        # Set Aspect Ratio resizing mode
        if self._arResizeMode == AspectRatioResizeMode.CROP:
            # Cropping is already the default mode of the ImageManip node
            self.imageManip.initialConfig.setResize(self._size)
        elif self._arResizeMode == AspectRatioResizeMode.LETTERBOX:
            self.imageManip.initialConfig.setResizeThumbnail(*self._size)
        elif self._arResizeMode == AspectRatioResizeMode.STRETCH:
            self.imageManip.initialConfig.setResize(self._size)
            self.imageManip.setKeepAspectRatio(False)  # Not keeping aspect ratio -> stretching the image

        return self.imageManip.out

    def config_multistage_nn(self,
                             debug=False,
                             labels: Optional[List[int]] = None,
                             scaleBb: Optional[Tuple[int, int]] = None,
                             ) -> None:
        """
        Configures the MultiStage NN pipeline. Available if the input to this NNComponent is Detection NNComponent.

        Args:
            debug (bool, default False): Debug script node
            labels (List[int], optional): Crop & run inference only on objects with these labels
            scaleBb (Tuple[int, int], optional): Scale detection bounding boxes (x, y) before cropping the frame. In %.
        """

        if not self._isMultiStage():
            print(
                "Input to this model was not a NNComponent, so 2-stage NN inferencing isn't possible! This configuration attempt will be ignored.")
            return

        self._multi_stage_config = MultiStageConfig(debug, labels, scaleBb)

    def _parse_label(self, label: Union[str, int]) -> int:
        if isinstance(label, int):
            return label
        elif isinstance(label, str):
            if not self._labels:
                raise ValueError(
                    "Incorrect trackLabels type! Make sure to pass NN configuration to the NNComponent so it can deccode string labels!")
            # Label map is Dict of either "name", or ["name", "color"]
            labelStrs = [l.upper() if isinstance(l, str) else l[0].upper() for l in self._labels]

            if label.upper() not in labelStrs: raise ValueError(f"String '{label}' wasn't found in passed labels!")
            return labelStrs.index(label.upper())
        else:
            raise Exception('_parse_label only accepts int or str')

    def config_tracker(self,
                       type: Optional[dai.TrackerType] = None,
                       trackLabels: Optional[List[int]] = None,
                       assignmentPolicy: Optional[dai.TrackerIdAssignmentPolicy] = None,
                       maxObj: Optional[int] = None,
                       threshold: Optional[float] = None
                       ):
        """
        Configure Object Tracker node (if it's enabled).

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
        if trackLabels and 0 < len(trackLabels):
            l = [self._parse_label(l) for l in trackLabels]
            self.tracker.setDetectionLabelsToTrack(l)
        if assignmentPolicy:
            self.tracker.setTrackerIdAssignmentPolicy(assignmentPolicy)
        if maxObj:
            if 60 < maxObj:
                raise ValueError("Maximum objects to track is 60!")
            self.tracker.setMaxObjectsToTrack(maxObj)
        if threshold:
            self.tracker.setTrackerThreshold(threshold)

    def config_yolo_from_metadata(self, metadata: Dict):
        """
        Configures (Spatial) Yolo Detection Network node with a dictionary. Calls config_yolo().
        """
        return self.config_yolo(
            numClasses=metadata['classes'],
            coordinateSize=metadata['coordinates'],
            anchors=metadata['anchors'],
            masks=metadata['anchor_masks'],
            iouThreshold=metadata['iou_threshold'],
            confThreshold=metadata['confidence_threshold'],
        )

    def config_yolo(self,
                    numClasses: int,
                    coordinateSize: int,
                    anchors: List[float],
                    masks: Dict[str, List[int]],
                    iouThreshold: float,
                    confThreshold: Optional[float] = None,
                    ) -> None:
        """
        Configures (Spatial) Yolo Detection Network node.

        """
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
                  confThreshold: Optional[float] = None,
                  aspectRatioResizeMode: AspectRatioResizeMode = None,
                  ):
        """
        Configures the Detection Network node.

        Args:
            confThreshold: (float, optional): Confidence threshold for the detections (0..1]
            aspectRatioResizeMode: (AspectRatioResizeMode, optional): Change aspect ratio resizing mode - to either STRETCH, CROP, or LETTERBOX
        """
        if aspectRatioResizeMode:
            self._arResizeMode = aspectRatioResizeMode
        if confThreshold and self._isDetector():
            self.node.setConfidenceThreshold(confThreshold)

    def config_spatial(self,
                       bbScaleFactor: Optional[float] = None,
                       lowerThreshold: Optional[int] = None,
                       upperThreshold: Optional[int] = None,
                       calcAlgo: Optional[dai.SpatialLocationCalculatorAlgorithm] = None,
                       ):
        """
        Configures the Spatial Detection Network node.

        Args:
            bbScaleFactor (float, optional): Specifies scale factor for detected bounding boxes (0..1]
            lowerThreshold (int, optional): Specifies lower threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            upperThreshold (int, optional): Specifies upper threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            calcAlgo (dai.SpatialLocationCalculatorAlgorithm, optional): Specifies spatial location calculator algorithm: Average/Min/Max
            out (Tuple[str, str], optional): Enable streaming depth + bounding boxes mappings to the host. Useful for debugging.
        """
        if not self._isSpatial():
            print('This is not a Spatial Detection network! This configuration attempt will be ignored.')
            return

        if bbScaleFactor:
            self.node.setBoundingBoxScaleFactor(bbScaleFactor)
        if lowerThreshold:
            self.node.setDepthLowerThreshold(lowerThreshold)
        if upperThreshold:
            self.node.setDepthUpperThreshold(upperThreshold)
        if calcAlgo:
            self.node.setSpatialCalculationAlgorithm(calcAlgo)

    """
    Available outputs (to the host) of this component
    """

    class Out:
        _comp: 'NNComponent'

        def __init__(self, nnComponent: 'NNComponent'):
            self._comp = nnComponent

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Default output. Streams NN results and high-res frames that were downscaled and used for inferencing.
            Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
            """

            if self._comp._isMultiStage():
                out = XoutTwoStage(self._comp._input, self._comp,
                                   self._comp._input._input.get_stream_xout(),  # CameraComponent
                                   StreamXout(self._comp._input.node.id, self._comp._input.node.out),
                                   # NnComponent (detections)
                                   StreamXout(self._comp.node.id, self._comp.node.out),
                                   # This NnComponent (2nd stage NN)
                                   )
            else:
                out = XoutNnResults(self._comp,
                                    self._comp._input.get_stream_xout(),  # CameraComponent
                                    StreamXout(self._comp.node.id, self._comp.node.out))  # NnComponent
            return self._comp._create_xout(pipeline, out)

        def passthrough(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Default output. Streams NN results and passthrough frames (frames used for inferencing)
            Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
            """
            if self._comp._isMultiStage():
                out = XoutTwoStage(self._comp._input, self._comp,
                                   StreamXout(self._comp._input.node.id, self._comp._input.node.passthrough),
                                   # Passthrough frame
                                   StreamXout(self._comp._input.node.id, self._comp._input.node.out),
                                   # NnComponent (detections)
                                   StreamXout(self._comp.node.id, self._comp.node.out),
                                   # This NnComponent (2nd stage NN)
                                   )
            else:
                out = XoutNnResults(self._comp,
                                    StreamXout(self._comp.node.id, self._comp.node.passthrough),
                                    StreamXout(self._comp.node.id, self._comp.node.out)
                                    )

            return self._comp._create_xout(pipeline, out)

        def spatials(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutSpatialBbMappings:
            """
            Streams depth and bounding box mappings (``SpatialDetectionNework.boundingBoxMapping``). Produces SpatialBbMappingPacket.
            """
            if not self._comp._isSpatial():
                raise Exception(
                    'SDK tried to output spatial data (depth + bounding box mappings), but this is not a Spatial Detection network!')

            out = XoutSpatialBbMappings(device,
                                        StreamXout(self._comp.node.id, self._comp.node.passthroughDepth),
                                        StreamXout(self._comp.node.id, self._comp.node.boundingBoxMapping)
                                        )
            return self._comp._create_xout(pipeline, out)

        def twostage_crops(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutFrames:
            """
            Streams 2. stage cropped frames to the host. Produces FramePacket.
            """
            if not self._comp._isMultiStage():
                raise Exception('SDK tried to output TwoStage crop frames, but this is not a Two-Stage NN component!')

            out = XoutFrames(StreamXout(self._comp._multiStageNn.manip.id, self._comp._multiStageNn.manip.out))
            return self._comp._create_xout(pipeline, out)

        def tracker(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutTracker:
            """
            Streams ObjectTracker tracklets and high-res frames that were downscaled and used for inferencing. Produces TrackerPacket.
            """
            if not self._comp._isTracker(): raise Exception(
                'Tracker was not enabled! Enable with cam.create_nn("[model]", tracker=True)!')
            self._comp.node.passthrough.link(self._comp.tracker.inputDetectionFrame)
            self._comp.node.out.link(self._comp.tracker.inputDetections)
            # TODO: add support for full frame tracking
            self._comp.node.passthrough.link(self._comp.tracker.inputTrackerFrame)

            out = XoutTracker(self._comp,
                              self._comp._input.get_stream_xout(),  # CameraComponent
                              StreamXout(self._comp.tracker.id, self._comp.tracker.out)
                              )
            return self._comp._create_xout(pipeline, out)

    out: Out

    # Checks
    def _isSpatial(self) -> bool:
        return self._spatial is not None  # todo fix if spatial is bool and equals to False

    def _isTracker(self) -> bool:
        # Currently, only object detectors are supported
        return self._isDetector() and self.tracker is not None

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

    def _isDetector(self) -> bool:
        """
        Currently these 2 object detectors are supported
        """
        return self._isYolo() or self._isMobileNet()

    def _isMultiStage(self):
        if not isinstance(self._input, type(self)):
            return False

        if not self._input._isDetector():
            raise Exception('Only object detector models can be used as an input to the NNComponent!')

        return True

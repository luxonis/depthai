import re

from .component import Component
from .camera_component import CameraComponent
from .stereo_component import StereoComponent
from .multi_stage_nn import MultiStageNN
from pathlib import Path
from typing import Callable, Optional, Union, List, Dict, Tuple
import depthai as dai
import json
import os
import blobconverter
from ..utils import loadModule, isUrl, getBlob, configPipeline


class NNComponent(Component):
    blob: dai.OpenVINO.Blob
    tracker: dai.node.ObjectTracker
    node: Union[
        None,
        dai.node.NeuralNetwork,
        dai.node.MobileNetDetectionNetwork,
        dai.node.MobileNetSpatialDetectionNetwork,
        dai.node.YoloDetectionNetwork,
        dai.node.YoloSpatialDetectionNetwork,
    ] = None
    manip: dai.node.ImageManip  # ImageManip used to resize the input to match the expected NN input size
    inputComponent: Optional[Component] = None  # Used for visualizer. Only set if component was passed as an input
    input: dai.Node.Output  # Original high-res input
    out: dai.Node.Output
    passthrough: dai.Node.Output
    size: Tuple[int, int]  # Input size to the NN
    _multiStageNn: MultiStageNN

    # For visualizer
    labels: List = None  # obj detector labels
    handler: Callable = None  # Custom model handler for decoding

    def __init__(self,
                 pipeline: dai.Pipeline,
                 model: Union[str, Path],  # str for SDK supported model or Path to custom model's json
                 input: Union[dai.Node.Output, Component],
                 out: Union[None, bool, str] = None,
                 nnType: Optional[str] = None,
                 name: Optional[str] = None,  # name of the node
                 tracker: bool = False,  # Enable object tracker - only for Object detection models
                 spatial: Union[None, bool, StereoComponent, dai.Node.Output] = None,
                 args=None  # User defined args
                 ) -> None:
        """
        Neural Network component that abstracts the following API nodes: NeuralNetwork, MobileNetDetectionNetwork,
        MobileNetSpatialDetectionNetwork, YoloDetectionNetwork, YoloSpatialDetectionNetwork, ObjectTracker
        (only for object detectors).

        Args:
            pipeline (dai.Pipeline)
            model (Union[str, Path]): str for SDK supported model / Path to blob or custom model's json
            nnType (str, optional): Type of the NN - Either Yolo or MobileNet
            input: (Union[Component, dai.Node.Output]): Input to the NN. If nn_component that is object detector, crop HQ frame at detections (Script node + ImageManip node)
            name (Optional[str]): Name of the node
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            out (bool, default False): Stream component's output to the host
        """
        super().__init__()
        self.input = input
        self.spatial = spatial

        # Parse the input config/model
        if isinstance(model, str):
            if isUrl(model):  # Download from the web
                model = getBlob(model)
            model = Path(model)
        conf = None

        if model.is_file():
            if model.suffix == '.blob':
                self.blob = dai.OpenVINO.Blob(model)
                # BlobConverter sets name of the blob '[name]_openvino_[version]_[num]cores.blob'
                # So we can parse this openvino version if it exists
                match = re.search('_openvino_\d{4}.\d', str(model))
                if match is not None:
                    version = match.group().replace('_openvino_', '')
                    configPipeline(pipeline, openvinoVersion=version)
            elif model.suffix == '.json':  # json config file was passed
                with model.open() as f:
                    conf = json.load(f)
        else:  # SDK supported model
            availableModels = self.getSupportedModels(printModels=False)
            if str(model) not in availableModels:
                raise ValueError(f"Specified model '{str(model)}' is not supported by DepthAI SDK. \
                    Check SDK documentation page to see which models are supproted.")

            model = availableModels[str(model)] / 'config.json'
            with model.open() as f:
                conf = json.load(f)

        if conf:
            self.blob = dai.OpenVINO.Blob(self.blob_from_config(conf['model'], model.parent))

            if "openvino_version" in conf:
                pipeline.setOpenVINOVersion(
                    getattr(dai.OpenVINO.Version, 'VERSION_' + self.conf.get("openvino_version")))

            nnConfig = conf.get("nn_config", {})
            if 'NN_family' in nnConfig:
                nnType = str(nnConfig['NN_family']).upper()

            # Save for visualization
            self.labels = conf.get("mappings", {}).get("labels", None)
            if 'handler' in conf:
                self.handler = loadModule(model.parent / conf["handler"])

                if not callable(getattr(self.handler, "decode", None)):
                    raise RuntimeError("Custom model handler does not contain 'decode' method!")

        nodeType = dai.node.NeuralNetwork
        if nnType:
            if nnType.upper() == 'YOLO':
                nodeType = dai.node.YoloSpatialDetectionNetwork if spatial else dai.node.YoloDetectionNetwork
            elif nnType.upper() == 'MOBILENET':
                nodeType = dai.node.MobileNetSpatialDetectionNetwork if spatial else dai.node.MobileNetDetectionNetwork

        self.node = pipeline.create(nodeType)
        self.node.setBlob(self.blob)
        self.out = self.node.out

        if conf:
            self.update_from_config(conf)

        # self.passthrough = self.node.passthrough

        if not self.node or not self.blob: raise NotImplementedError()

        if 1 < len(self.blob.networkInputs):
            raise NotImplementedError()

        nnIn: dai.TensorInfo = next(iter(self.blob.networkInputs.values()))
        # TODO: support models that expect mono img
        self.size: Tuple[int, int] = (nnIn.dims[0], nnIn.dims[1])
        # maxSize = dims

        if isinstance(input, CameraComponent):
            self.inputComponent = input
            self.input = input.out
            self._createResizeManip(pipeline, self.size, input.out).link(self.node.input)
        elif isinstance(input, type(self)):
            if not input.isDetector():
                raise Exception('Only object detector models can be used as an input to the NNComponent!')
            self.inputComponent = input  # Used by visualizer

            # Create script node, get HQ frames from input.
            self._multiStageNn = MultiStageNN(pipeline, input, input.input, self.size)
            self._multiStageNn.out.link(self.node.input)  # Cropped frames
            # For debugging, for intenral counter
            self.node.out.link(self._multiStageNn.script.inputs['recognition'])
            self.node.input.setBlocking(True)
            self.node.input.setQueueSize(15)
        elif isinstance(input, dai.Node.Output):
            # Link directly via ImageManip
            self.input = input
            self._createResizeManip(pipeline, self.size, input).link(self.node.input)

        if spatial:
            if isinstance(spatial, bool):  # Create new StereoComponent
                spatial = StereoComponent(pipeline, args=args)
            if isinstance(spatial, StereoComponent):
                spatial.depth.link(self.node.inputDepth)
            elif isinstance(spatial, dai.Node.Output):
                spatial.link(self.node.inputDepth)

        if tracker:
            if not self.isDetector():
                print('Currently, only object detector models (Yolo/MobileNet) can use tracker!')
            else:
                raise NotImplementedError()
                # self.tracker = pipeline.createObjectTracker()
                # self.out = self.tracker.out

        if out:
            super().createXOut(
                pipeline,
                type(self),
                name=out,
                out=self.out,
                depthaiMsg=dai.ImgDetections if self.isDetector() else dai.NNData
            )

    def blob_from_config(self, model, path: Path) -> str:
        if 'blob' in model:
            return str(path / model['blob'])

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

        raise ValueError("Specified `model` values in NN config files are incorrect!")

    def _createResizeManip(self,
                           pipeline: dai.Pipeline,
                           size: Tuple[int, int],
                           input: dai.Node.Output) -> dai.Node.Output:
        """
        Creates manip that resizes the input to match the expected NN input size
        """
        self.manip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(size)
        self.manip.setMaxOutputFrameSize(size[0] * size[1] * 3)
        self.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        input.link(self.manip.inputImage)
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
        return self.configYolo(
            numClasses=metadata['classes'],
            coordinateSize=metadata['coordinates'],
            anchors=metadata['anchors'],
            masks=metadata['anchor_masks'],
            iouThreshold=metadata['iou_threshold'],
            confThreshold=metadata['confidence_threshold'],
        )

    def configYolo(self,
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

    def update_from_config(self, conf: Union[Path, Dict]) -> None:
        """
        Updates the NN component from the config file.
        @param conf: Config file, either path to json file, or dictionary
        """
        if isinstance(conf, Path):
            with open(conf.resolve(), 'r') as f:
                conf = json.loads(f.read())

        # Configure node based on conf file
        nnConfig = conf.get("nn_config", {})
        if self.isDetector() and 'confidence_threshold' in nnConfig:
            self.node.setConfidenceThreshold(int(nnConfig['confidence_threshold']))

        meta = nnConfig.get('NN_specific_metadata', None)
        if self.isYolo() and meta:
            self.configYoloFromMeta(metadata=meta)

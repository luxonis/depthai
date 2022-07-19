from .component import Component
from.camera_component import CameraComponent
from .stereo_component import StereoComponent
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import depthai as dai

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
    ]
    _input: Union[Component, dai.Node.Output]
    manip: dai.node.ImageManip
    out: dai.Node.Output
    resizeManip: dai.node.ImageManip # ImageManip used to resize the input to match the expected NN input size

    def __init__(self,
        pipeline: dai.Pipeline,
        model: Union[str, Path], # str for SDK supported model or Path to custom model's json
        input: Union[Component, dai.Node.Output],
        nnType: Optional[str] = None,
        name: Optional[str] = None, # name of the node
        tracker: bool = False, # Enable object tracker - only for Object detection models
        spatial: Union[None, bool, StereoComponent, dai.Node.Output] = None,
        out: bool = False,
        args = None # User defined args
        ) -> None:
        """
        Neural Network component that abstracts the following API nodes: NeuralNetwork, MobileNetDetectionNetwork,
        MobileNetSpatialDetectionNetwork, YoloDetectionNetwork, YoloSpatialDetectionNetwork, ObjectTracker
        (only for object detectors).

        Args:
            pipeline (dai.Pipeline)
            model (Union[str, Path]): str for SDK supported model / Path to blob or custom model's json
            type (str, optional): Type of the NN - Either Yolo or MobileNet
            input: (Union[Component, dai.Node.Output]): Input to the NN. If nn_component that is object detector, crop HQ frame at detections (Script node + ImageManip node)
            name (Optional[str]): Name of the node
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            out (bool, default False): Stream component's output to the host
        """
        self._input = input

        # TODO: parse config / use blobconverter to download the model
        if isinstance(model, str):
            path = Path(model)
            if path.is_file():
                self._parsePath(path)
            else:
                # SDK supported model OR download from model zoo
                pass
        elif isinstance(model, Path):
            self._parsePath(path)
        else:
            raise Exception("model must be Path/str")

        if nnType:
            if nnType.upper() == 'YOLO':
                nodeType = dai.node.YoloSpatialDetectionNetwork if spatial else dai.node.YoloDetectionNetwork
            elif nnType.upper() == 'MOBILENET':
                nodeType = dai.node.MobileNetSpatialDetectionNetwork if spatial else dai.node.MobileNetDetectionNetwork
            else:
                nodeType = dai.node.NeuralNetwork

            self.node = pipeline.create(nodeType)
            self.node.setBlob(self.blob)
            self.out = self.node.out

        
        # TODO implement reading from config
        if not self.node or not self.blob: raise NotImplementedError()

        if 1 < len(self.blob.networkInputs):
            raise NotImplementedError()
        
        nnIn: dai.TensorInfo = next(iter(self.blob.networkInputs.values()))
        # TODO: support models that expect mono img
        size: Tuple[int, int] = (nnIn.dims[0], nnIn.dims[1])
        # maxSize = dims
        
        if isinstance(input, CameraComponent):
            self._createResizeManip(pipeline, size, input.out).link(self.node.input)
        elif isinstance(input, type(self)):
            if not input.isDetector():
                raise Exception('Only object detector models can be used as an input to the NNComponent!')
            input.isMobileNet
            # Create script node, get HQ frames from input.

            raise NotImplementedError()
        elif isinstance(input, dai.Node.Output):
            # Link directly via ImageManip
            self._createResizeManip(pipeline, size, input).link(self.node.input)

        if spatial:
            if isinstance(spatial, bool): # Create new StereoComponent
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
                self.tracker = pipeline.createObjectTracker()


    def _createResizeManip(self, pipeline: dai.Pipeline, size: Tuple[int, int], input: dai.Node.Output) -> dai.Node.Output:
        """
        Creates manip that resizes the input to match the expected NN input size
        """
        self.resizeManip = pipeline.create(dai.node.ImageManip)
        self.resizeManip.initialConfig.setResize(size)
        self.resizeManip.setMaxOutputFrameSize(size[0] * size[1] *3)
        input.link(self.resizeManip.inputImage)
        return self.resizeManip.out

    def _parsePath(self, path: Path) -> None:
        if path.suffix == '.blob':
            self.blob = dai.OpenVINO.Blob(path)
        elif path.suffix == '.json':
            # TODO: Parse json config file
            pass

    def configCropping(self,
        labels: Optional[List[int]] = None,
    ) -> None:
        """
        For multi-stage NN pipelines. Available if the input to this NNComponent was another NN component.

        Args:
            labels (Optional[List[int]]): Crop & run inference only on objects with these labels

        """
        if not isinstance(self._input, type(self)):
            print("Input to this model was not a NNComponent, so 2-stage NN inferencing isn't possible! This configuration attempt will be ignored.")
            return

        raise NotImplementedError()

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
            print("Tracker was not enabled! Enable with cam.create_nn('[model]', tracker=True). This configuration attempt will be ignored.")
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

    def configYolo(self,
        numClasses: int,
        coordinateSize: int,
        anchors: List[float],
        masks: Dict[str, List[int]],
        iouThreshold: float,
        confThreshold: Optional[float] = None,
        config: Optional[Dict] = None
    ) -> None:
        if not self.isYolo():
            print('This is not a YOLO detection network! This configuration attempt will be ignored.')
            return

        if config: # Metadata from eg. tools.luxonis.com
            return self.configYolo(
                numClasses=config['classes'],
                coordinateSize=config['classes'],
                anchors=config['anchors'],
                masks=config['anchor_masks'],
                iouThreshold=config['iou_threshold'],
                confThreshold=config['confidence_threshold'],
            )

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
        calcAlgo: Optional[dai.SpatialLocationCalculatorAlgorithm] = None
    ) -> None:
        if not self.isSpatial():
            print('This is not a Spatial Detection network! This configuration attempt will be ignored.')
            return

        if bbScaleFactor: self.node.setBoundingBoxScaleFactor(bbScaleFactor)
        if lowerThreshold: self.node.setDepthLowerThreshold(lowerThreshold)
        if upperThreshold: self.node.setDepthUpperThreshold(upperThreshold)
        if calcAlgo: self.node.setSpatialCalculationAlgorithm(calcAlgo)


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

import json
import warnings
from pathlib import Path
from typing import Callable, Union, List, Dict

try:
    import blobconverter
except ImportError:
    blobconverter = None

from depthai_sdk.classes.nn_config import Config
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.components.integrations.roboflow import RoboflowIntegration
from depthai_sdk.components.multi_stage_nn import MultiStageNN, MultiStageConfig
from depthai_sdk.components.nn_helper import *
from depthai_sdk.components.parser import *
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout, XoutBase
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_nn import XoutTwoStage, XoutNnResults, XoutSpatialBbMappings
from depthai_sdk.oak_outputs.xout.xout_nn_encoded import XoutNnMjpeg, XoutNnH26x
from depthai_sdk.oak_outputs.xout.xout_tracker import XoutTracker
from depthai_sdk.replay import Replay


class NNComponent(Component):
    def __init__(self,
                 pipeline: dai.Pipeline,
                 model: Union[str, Path, Dict],  # str for SDK supported model or Path to custom model's json
                 input: Union[CameraComponent, 'NNComponent'],
                 nn_type: Optional[str] = None,  # Either 'yolo' or 'mobilenet'
                 decode_fn: Optional[Callable] = None,
                 tracker: bool = False,  # Enable object tracker - only for Object detection models
                 spatial: Union[None, bool, StereoComponent] = None,
                 replay: Optional[Replay] = None,
                 args: Dict = None,  # User defined args
                 name: Optional[str] = None
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
            nn_type (str, optional): Type of the NN - Either 'Yolo' or 'MobileNet'
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            replay (Replay object): Replay
            args (Any, optional): Use user defined arguments when constructing the pipeline
            name (str, optional): Name of the output stream
        """
        super().__init__()

        self.name = name
        self.out = self.Out(self)
        self.node: Optional[
            dai.node.NeuralNetwork,
            dai.node.MobileNetDetectionNetwork,
            dai.node.MobileNetSpatialDetectionNetwork,
            dai.node.YoloDetectionNetwork,
            dai.node.YoloSpatialDetectionNetwork] = None

        # ImageManip used to resize the input to match the expected NN input size
        self.image_manip: Optional[dai.node.ImageManip] = None
        self.x_in: Optional[dai.node.XLinkIn] = None  # Used for multi-stage pipeline
        self.tracker = pipeline.createObjectTracker() if tracker else None

        # Private properties
        self._ar_resize_mode: ResizeMode = ResizeMode.LETTERBOX  # Default
        self._input: Union[CameraComponent, 'NNComponent']  # Input to the NNComponent node passed on initialization
        self._stream_input: dai.Node.Output  # Node Output that will be used as the input for this NNComponent

        self._blob: Optional[dai.OpenVINO.Blob] = None
        self._forced_version: Optional[dai.OpenVINO.Version] = None  # Forced OpenVINO version
        self._size: Optional[Tuple[int, int]] = None  # Input size to the NN
        self._args: Optional[Dict] = None
        self._config: Optional[Dict] = None
        self._node_type: dai.node = dai.node.NeuralNetwork  # Type of the node for `node`
        self._roboflow: Optional[RoboflowIntegration] = None

        self._multi_stage_nn: Optional[MultiStageNN] = None
        self._multi_stage_config: Optional[MultiStageConfig] = None

        self._input_queue = Optional[None]  # Input queue for multi-stage pipeline

        self._spatial: Optional[Union[bool, StereoComponent]] = None
        self._replay: Optional[Replay]  # Replay module

        # For visualizer
        self._labels: Optional[List] = None  # Obj detector labels
        self._handler: Optional[Callable] = None  # Custom model handler for decoding

        # Save passed settings
        self._input = input
        self._spatial = spatial
        self._args = args
        self._replay = replay
        self._decode_fn = decode_fn or None  # Decode function that will be used to decode NN results

        # Parse passed settings
        self._parse_model(model)
        if nn_type:
            self._parse_node_type(nn_type)

        # Create NN node
        self.node = pipeline.create(self._node_type)
        self._update_config()

    def forced_openvino_version(self) -> dai.OpenVINO.Version:
        """
        Checks whether the component forces a specific OpenVINO version. This function is called after
        Camera has been configured and right before we connect to the OAK camera.
        @return: Forced OpenVINO version (optional).
        """
        return self._forced_version

    def on_init(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        if self._roboflow:
            path = self._roboflow.device_update(device)
            self._parse_config(path)
            self._update_config()

        if self._blob is None:
            self._blob = dai.OpenVINO.Blob(self._blob_from_config(self._config['model'], version))

        # TODO: update NN input based on camera resolution
        self.node.setBlob(self._blob)
        self._out = self.node.out

        if 1 < len(self._blob.networkInputs):
            raise NotImplementedError()

        nn_in: dai.TensorInfo = next(iter(self._blob.networkInputs.values()))
        # TODO: support models that expect mono img
        self._size: Tuple[int, int] = (nn_in.dims[0], nn_in.dims[1])
        # maxSize = dims

        if isinstance(self._input, CameraComponent):
            self._stream_input = self._input.stream
            self._setup_resize_manip(pipeline).link(self.node.input)
        elif self._is_multi_stage():
            # Calculate crop shape of the object detector
            frame_size = self._input._input.stream_size
            nn_size = self._input._size
            scale = frame_size[0] / nn_size[0], frame_size[1] / nn_size[1]
            i = 0 if scale[0] < scale[1] else 1
            crop = int(scale[i] * nn_size[0]), int(scale[i] * nn_size[1])
            # Crop the high-resolution frames, so it matches object detection frame aspect ratio

            self.image_manip = pipeline.createImageManip()
            self.image_manip.setNumFramesPool(10)
            self.image_manip_config = dai.ImageManipConfig()
            self.image_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

            self._input._stream_input.link(self.image_manip.inputImage)
            if self._input._is_detector() and self._decode_fn is None:
                self.image_manip.setResize(*crop)
                self.image_manip.setMaxOutputFrameSize(crop[0] * crop[1] * 3)

                # Create script node, get HQ frames from input.
                self._multi_stage_nn = MultiStageNN(pipeline, self._input.node, self.image_manip.out, self._size)
                self._multi_stage_nn.configure(self._multi_stage_config)
                self._multi_stage_nn.out.link(self.node.input)  # Cropped frames

                # For debugging, for integral counter
                self.node.out.link(self._multi_stage_nn.script.inputs['recognition'])
                self.node.input.setBlocking(True)
                self.node.input.setQueueSize(20)
            else:
                print('Using on-host decoding for multi-stage NN')
                # Custom NN
                self.image_manip.setResize(*self._size)
                self.image_manip.setMaxOutputFrameSize(self._size[0] * self._size[1] * 3)

                # TODO pass frame on device, and just send config from host
                self.x_in = pipeline.createXLinkIn()
                self.x_in.setStreamName("input_queue")
                self.x_in.setMaxDataSize(frame_size[0] * frame_size[1] * 3)
                self.x_in.out.link(self.image_manip.inputImage)

                self.x_in_cfg = pipeline.createXLinkIn()
                self.x_in_cfg.setStreamName("input_queue_cfg")
                self.x_in_cfg.out.link(self.image_manip.inputConfig)

                self.image_manip.out.link(self.node.input)
                self.node.input.setQueueSize(20)
        else:
            raise ValueError(
                "'input' argument passed on init isn't supported!"
                "You can only use NnComponent or CameraComponent as the input."
            )

        if self._spatial:
            if isinstance(self._spatial, bool):  # Create new StereoComponent
                self._spatial = StereoComponent(pipeline, args=self._args, replay=self._replay)
                self._spatial.on_init(pipeline, device, version)
            if isinstance(self._spatial, StereoComponent):
                self._spatial.depth.link(self.node.inputDepth)
                self._spatial.config_stereo(align=self._input._source)
            # Configure Spatial Detection Network

        if self._args:
            if self._is_spatial():
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
                self._forced_version = self._blob.version
            elif model.suffix == '.json':  # json config file was passed
                self._parse_config(model)
        else:  # SDK supported model
            models = getSupportedModels(printModels=False)
            zoo_models = blobconverter.zoo_list()

            if str(model) in models:
                model = models[str(model)] / 'config.json'
                self._parse_config(model)
            elif str(model) in zoo_models:
                print(
                    'Models from the OpenVINO Model Zoo do not carry any metadata'
                    ' (e.g., label map, decoding logic). Please keep this in mind when using models from Zoo.'
                )
                self._blob = dai.OpenVINO.Blob(blobconverter.from_zoo(str(model), shaves=6))
                self._forced_version = self._blob.version
            else:
                raise ValueError(f"Specified model '{str(model)}' is not supported by DepthAI SDK.\n"
                                 "Check SDK documentation page to see which models are supported.")

    def _parse_node_type(self, nn_type: str) -> None:
        self._node_type = dai.node.NeuralNetwork
        if nn_type:
            if nn_type.upper() == 'YOLO':
                self._node_type = dai.node.YoloSpatialDetectionNetwork if self._is_spatial() else dai.node.YoloDetectionNetwork
            elif nn_type.upper() == 'MOBILENET':
                self._node_type = dai.node.MobileNetSpatialDetectionNetwork if self._is_spatial() else dai.node.MobileNetDetectionNetwork

    def _config_spatials_args(self, args):
        if not isinstance(args, Dict):
            args = vars(args)  # Namespace -> Dict
        self.config_spatial(
            bb_scale_factor=args.get('sbbScaleFactor', None),
            lower_threshold=args.get('minDepth', None),
            upper_threshold=args.get('maxDepth', None),
        )

    def _parse_config(self, model_config: Union[Path, str, Dict]):
        """
        Called when NNComponent is initialized. Reads config.json file and parses relevant setting from there
        """
        parent_folder = None
        if isinstance(model_config, str):
            model_config = Path(model_config).resolve()

        if isinstance(model_config, Path):
            parent_folder = model_config.parent
            with model_config.open() as f:
                self._config = Config().load(json.loads(f.read()))
        else:  # Dict
            self._config = model_config

        if 'source' in self._config:
            if self._config['source'] == 'roboflow':
                from depthai_sdk.components.integrations.roboflow import RoboflowIntegration
                self._roboflow = RoboflowIntegration(self._config)
                self._parse_node_type('YOLO')  # Roboflow only supports YOLO models
                return
            else:
                raise ValueError(f"[NN Dict configuration] Source '{self._config['source']}' not supported")

        # Get blob from the config file
        if 'model' in self._config:
            model = self._config['model']

            # Resolve the paths inside config
            if parent_folder:
                for name in ['blob', 'xml', 'bin']:
                    if name in model:
                        model[name] = str((parent_folder / model[name]).resolve())

            if 'blob' in model:
                self._blob = dai.OpenVINO.Blob(model['blob'])

        # Parse OpenVINO version
        if "openvino_version" in self._config:
            self._forced_version = parse_open_vino_version(self._config.get("openvino_version"))

        # Save for visualization
        self._labels = self._config.get("mappings", {}).get("labels", None)

        # Handler.py logic to decode raw NN results into standardized AI results
        if 'handler' in self._config:
            self._handler = loadModule(model_config.parent / self._config["handler"])

            if not callable(getattr(self._handler, "decode", None)):
                raise RuntimeError("Custom model handler does not contain 'decode' method!")

        if 'nn_config' in self._config:
            nn_config = self._config.get("nn_config", {})

            # Parse node type
            nn_family = nn_config.get("NN_family", None)
            if nn_family:
                self._parse_node_type(nn_family)

    def _blob_from_config(self, model: Dict, version: dai.OpenVINO.Version) -> str:
        """
        Gets the blob from the config file.
        """
        vals = str(version).split('_')
        version_str = f"{vals[1]}.{vals[2]}"

        if 'model_name' in model:  # Use blobconverter to download the model
            zoo_type = model.get("zoo", 'intel')
            return blobconverter.from_zoo(model['model_name'],
                                          zoo_type=zoo_type,
                                          shaves=6,  # TODO: Calculate ideal shave amount
                                          version=version_str
                                          )

        if 'xml' in model and 'bin' in model:
            return blobconverter.from_openvino(xml=model['xml'],
                                               bin=model['bin'],
                                               data_type="FP16",  # Myriad X
                                               shaves=6,  # TODO: Calculate ideal shave amount
                                               version=version_str
                                               )

        raise ValueError("Specified `model` values in json config files are incorrect!")

    def _setup_resize_manip(self, pipeline: Optional[dai.Pipeline] = None) -> dai.Node.Output:
        """
        Creates ImageManip node that resizes the input to match the expected NN input size.
        DepthAI uses CHW (Planar) channel layout and BGR color order convention.
        """
        if not self.image_manip:
            self.image_manip = pipeline.create(dai.node.ImageManip)
            self._stream_input.link(self.image_manip.inputImage)
            self.image_manip.setMaxOutputFrameSize(self._size[0] * self._size[1] * 3)
            self.image_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            # Set to non-blocking
            self.image_manip.inputImage.setBlocking(False)
            self.image_manip.inputImage.setQueueSize(2)

        # Set Aspect Ratio resizing mode
        if self._ar_resize_mode == ResizeMode.CROP:
            # Cropping is already the default mode of the ImageManip node
            self.image_manip.initialConfig.setResize(self._size)
        elif self._ar_resize_mode == ResizeMode.LETTERBOX:
            self.image_manip.initialConfig.setResizeThumbnail(*self._size)
        elif self._ar_resize_mode == ResizeMode.STRETCH:
            self.image_manip.initialConfig.setResize(self._size)
            self.image_manip.setKeepAspectRatio(False)  # Not keeping aspect ratio -> stretching the image

        return self.image_manip.out

    def config_multistage_nn(self,
                             debug=False,
                             labels: Optional[List[int]] = None,
                             scale_bb: Optional[Tuple[int, int]] = None,
                             ) -> None:
        """
        Configures the MultiStage NN pipeline. Available if the input to this NNComponent is Detection NNComponent.

        Args:
            debug (bool, default False): Debug script node
            labels (List[int], optional): Crop & run inference only on objects with these labels
            scale_bb (Tuple[int, int], optional): Scale detection bounding boxes (x, y) before cropping the frame. In %.
        """
        if not self._is_multi_stage():
            print("Input to this model was not a NNComponent, so 2-stage NN inferencing isn't possible!"
                  "This configuration attempt will be ignored.")
            return

        self._multi_stage_config = MultiStageConfig(debug, labels, scale_bb)

    def _parse_label(self, label: Union[str, int]) -> int:
        if isinstance(label, int):
            return label
        elif isinstance(label, str):
            if not self._labels:
                raise ValueError("Incorrect trackLabels type! Make sure to pass NN configuration to"
                                 "the NNComponent so it can deccode string labels!")
            # Label map is Dict of either "name", or ["name", "color"]
            label_strs = [l.upper() if isinstance(l, str) else l[0].upper() for l in self._labels]

            if label.upper() not in label_strs: raise ValueError(f"String '{label}' wasn't found in passed labels!")
            return label_strs.index(label.upper())
        else:
            raise Exception('_parse_label only accepts int or str')

    def config_tracker(self,
                       tracker_type: Optional[dai.TrackerType] = None,
                       track_labels: Optional[List[int]] = None,
                       assignment_policy: Optional[dai.TrackerIdAssignmentPolicy] = None,
                       max_obj: Optional[int] = None,
                       threshold: Optional[float] = None
                       ):
        """
        Configure Object Tracker node (if it's enabled).

        Args:
            tracker_type (dai.TrackerType, optional): Set object tracker type
            track_labels (List[int], optional): Set detection labels to track
            assignment_policy (dai.TrackerType, optional): Set object tracker ID assignment policy
            max_obj (int, optional): Set max objects to track. Max 60.
            threshold (float, optional): Set threshold for object detection confidence. Default: 0.0
        """

        if self.tracker is None:
            warnings.warn("Tracker was not enabled! Enable with cam.create_nn('[model]', tracker=True)."
                          "This configuration attempt will be ignored.")
            return

        if tracker_type:
            self.tracker.setTrackerType(type=tracker_type)

        if track_labels and 0 < len(track_labels):
            labels = [self._parse_label(l) for l in track_labels]
            self.tracker.setDetectionLabelsToTrack(labels)

        if assignment_policy:
            self.tracker.setTrackerIdAssignmentPolicy(assignment_policy)

        if max_obj:
            if 60 < max_obj:
                raise ValueError("Maximum objects to track is 60!")
            self.tracker.setMaxObjectsToTrack(max_obj)

        if threshold:
            self.tracker.setTrackerThreshold(threshold)

    def config_yolo_from_metadata(self, metadata: Dict):
        """
        Configures (Spatial) Yolo Detection Network node with a dictionary. Calls config_yolo().
        """
        return self.config_yolo(
            num_classes=metadata['classes'],
            coordinate_size=metadata['coordinates'],
            anchors=metadata['anchors'],
            masks=metadata['anchor_masks'],
            iou_threshold=metadata['iou_threshold'],
            conf_threshold=metadata['confidence_threshold'],
        )

    def config_yolo(self,
                    num_classes: int,
                    coordinate_size: int,
                    anchors: List[float],
                    masks: Dict[str, List[int]],
                    iou_threshold: float,
                    conf_threshold: Optional[float] = None,
                    ) -> None:
        """
        Configures (Spatial) Yolo Detection Network node.

        """
        if not self._is_yolo():
            print('This is not a YOLO detection network! This configuration attempt will be ignored.')
            return

        if not self.node:
            raise Exception('YOLO node not initialized!')

        self.node.setNumClasses(num_classes)
        self.node.setCoordinateSize(coordinate_size)
        self.node.setAnchors(anchors)
        self.node.setAnchorMasks(masks)
        self.node.setIouThreshold(iou_threshold)

        if conf_threshold:
            self.node.setConfidenceThreshold(conf_threshold)

    def config_nn(self,
                  conf_threshold: Optional[float] = None,
                  resize_mode: Union[ResizeMode, str] = None):
        """
        Configures the Detection Network node.

        Args:
            conf_threshold: (float, optional): Confidence threshold for the detections (0..1]
            resize_mode: (ResizeMode, optional): Change aspect ratio resizing mode - to either STRETCH, CROP, or LETTERBOX.
        """
        if resize_mode:
            if isinstance(resize_mode, str):
                try:
                    resize_mode = ResizeMode[resize_mode.upper()]
                except (AttributeError, KeyError):
                    print('AR resize mode was not recognizied.'
                          'Options (case insensitive): STRETCH, CROP, LETTERBOX.'
                          'Using default LETTERBOX mode.')

            self._ar_resize_mode = resize_mode
        if conf_threshold and self._is_detector():
            self.node.setConfidenceThreshold(conf_threshold)

    def config_spatial(self,
                       bb_scale_factor: Optional[float] = None,
                       lower_threshold: Optional[int] = None,
                       upper_threshold: Optional[int] = None,
                       calc_algo: Optional[dai.SpatialLocationCalculatorAlgorithm] = None):
        """
        Configures the Spatial Detection Network node.

        Args:
            bb_scale_factor (float, optional): Specifies scale factor for detected bounding boxes (0..1]
            lower_threshold (int, optional): Specifies lower threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            upper_threshold (int, optional): Specifies upper threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            calc_algo (dai.SpatialLocationCalculatorAlgorithm, optional): Specifies spatial location calculator algorithm: Average/Min/Max
        """
        if not self._is_spatial():
            print('This is not a Spatial Detection network! This configuration attempt will be ignored.')
            return

        if bb_scale_factor:
            self.node.setBoundingBoxScaleFactor(bb_scale_factor)
        if lower_threshold:
            self.node.setDepthLowerThreshold(lower_threshold)
        if upper_threshold:
            self.node.setDepthUpperThreshold(upper_threshold)
        if calc_algo:
            self.node.setSpatialCalculationAlgorithm(calc_algo)

    def _update_config(self):
        if self.node is None or self._config is None:
            return

        nn_config = self._config.get("nn_config", {})

        meta = nn_config.get('NN_specific_metadata', None)
        if self._is_yolo() and meta:
            self.config_yolo_from_metadata(metadata=meta)

        self.config_nn(conf_threshold=nn_config.get('conf_threshold', None))

    """
    Available outputs (to the host) of this component
    """

    class Out:
        def __init__(self, nn_component: 'NNComponent'):
            self._comp = nn_component

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Default output. Streams NN results and high-res frames that were downscaled and used for inferencing.
            Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
            """

            if self._comp._is_multi_stage():
                det_nn_out = StreamXout(id=self._comp._input.node.id,
                                        out=self._comp._input.node.out,
                                        name=self._comp._input.name)
                second_nn_out = StreamXout(id=self._comp.node.id, out=self._comp.node.out, name=self._comp.name)

                out = XoutTwoStage(det_nn=self._comp._input,
                                   second_nn=self._comp,
                                   frames=self._comp._input._input.get_stream_xout(),
                                   det_out=det_nn_out,
                                   second_nn_out=second_nn_out,
                                   device=device,
                                   input_queue_name="input_queue" if self._comp.x_in else None)
            else:
                det_nn_out = StreamXout(id=self._comp.node.id, out=self._comp.node.out, name=self._comp.name)

                out = XoutNnResults(det_nn=self._comp,
                                    frames=self._comp._input.get_stream_xout(),
                                    nn_results=det_nn_out)

            return self._comp._create_xout(pipeline, out)

        def passthrough(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Default output. Streams NN results and passthrough frames (frames used for inferencing)
            Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
            """
            if self._comp._is_multi_stage():
                det_nn_out = StreamXout(id=self._comp._input.node.id,
                                        out=self._comp._input.node.out,
                                        name=self._comp._input.name)
                frames = StreamXout(id=self._comp._input.node.id,
                                    out=self._comp._input.node.passthrough,
                                    name=self._comp.name)
                second_nn_out = StreamXout(self._comp.node.id, self._comp.node.out, name=self._comp.name)

                out = XoutTwoStage(det_nn=self._comp._input,
                                   second_nn=self._comp,
                                   frames=frames,
                                   det_out=det_nn_out,
                                   second_nn_out=second_nn_out,
                                   device=device,
                                   input_queue_name="input_queue" if self._comp.x_in else None)
            else:
                det_nn_out = StreamXout(id=self._comp.node.id, out=self._comp.node.out, name=self._comp.name)
                frames = StreamXout(id=self._comp.node.id, out=self._comp.node.passthrough, name=self._comp.name)

                out = XoutNnResults(det_nn=self._comp,
                                    frames=frames,
                                    nn_results=det_nn_out)

            return self._comp._create_xout(pipeline, out)

        def image_manip(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            out = XoutFrames(frames=StreamXout(id=self._comp.image_manip.id,
                                               out=self._comp.image_manip.out,
                                               name=self._comp.name))
            return self._comp._create_xout(pipeline, out)

        def input(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            out = XoutFrames(frames=StreamXout(id=self._comp._input.node.id,
                                               out=self._comp._stream_input,
                                               name=self._comp.name))
            return self._comp._create_xout(pipeline, out)

        def spatials(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutSpatialBbMappings:
            """
            Streams depth and bounding box mappings (``SpatialDetectionNework.boundingBoxMapping``). Produces SpatialBbMappingPacket.
            """
            if not self._comp._is_spatial():
                raise Exception('SDK tried to output spatial data (depth + bounding box mappings),'
                                'but this is not a Spatial Detection network!')

            out = XoutSpatialBbMappings(
                device=device,
                frames=StreamXout(id=self._comp.node.id, out=self._comp.node.passthroughDepth, name=self._comp.name),
                configs=StreamXout(id=self._comp.node.id, out=self._comp.node.out, name=self._comp.name)
            )

            return self._comp._create_xout(pipeline, out)

        def twostage_crops(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutFrames:
            """
            Streams 2. stage cropped frames to the host. Produces FramePacket.
            """
            if not self._comp._is_multi_stage():
                raise Exception('SDK tried to output TwoStage crop frames, but this is not a Two-Stage NN component!')

            out = XoutFrames(frames=StreamXout(id=self._comp._multi_stage_nn.manip.id,
                                               out=self._comp._multi_stage_nn.manip.out,
                                               name=self._comp.name))

            return self._comp._create_xout(pipeline, out)

        def tracker(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutTracker:
            """
            Streams ObjectTracker tracklets and high-res frames that were downscaled and used for inferencing. Produces TrackerPacket.
            """
            if not self._comp._is_tracker():
                raise Exception('Tracker was not enabled! Enable with cam.create_nn("[model]", tracker=True)!')

            self._comp.node.passthrough.link(self._comp.tracker.inputDetectionFrame)
            self._comp.node.out.link(self._comp.tracker.inputDetections)

            # TODO: add support for full frame tracking
            self._comp.node.passthrough.link(self._comp.tracker.inputTrackerFrame)

            out = XoutTracker(det_nn=self._comp,
                              frames=self._comp._input.get_stream_xout(),  # CameraComponent
                              tracklets=StreamXout(self._comp.tracker.id, self._comp.tracker.out))

            return self._comp._create_xout(pipeline, out)

        def encoded(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutNnResults:
            """
            Streams NN results and encoded frames (frames used for inferencing)
            Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
            """
            if self._comp._input.encoder is None:
                raise Exception('Encoder not enabled for the input')

            if self._comp._is_multi_stage():
                raise NotImplementedError('Encoded output not supported for 2-stage NNs at the moment.')

            if self._comp._input._encoder_profile == dai.VideoEncoderProperties.Profile.MJPEG:
                out = XoutNnMjpeg(
                    det_nn=self._comp,
                    frames=StreamXout(self._comp._input.encoder.id, self._comp._input.encoder.bitstream),
                    nn_results=StreamXout(self._comp.node.id, self._comp.node.out),
                    color=self._comp._input.is_color(),
                    lossless=self._comp._input.encoder.getLossless(),
                    fps=self._comp._input.encoder.getFrameRate(),
                    frame_shape=self._comp._input.stream_size
                )
            else:
                out = XoutNnH26x(
                    det_nn=self._comp,
                    frames=StreamXout(self._comp._input.node.id, self._comp._input.encoder.bitstream),
                    nn_results=StreamXout(self._comp.node.id, self._comp.node.out),
                    color=self._comp._input.is_color(),
                    profile=self._comp._input._encoder_profile,
                    fps=self._comp._input.encoder.getFrameRate(),
                    frame_shape=self._comp._input.stream_size
                )

            return self._comp._create_xout(pipeline, out)

    # Checks
    def _is_spatial(self) -> bool:
        return self._spatial is not None  # todo fix if spatial is bool and equals to False

    def _is_tracker(self) -> bool:
        # Currently, only object detectors are supported
        return self._is_detector() and self.tracker is not None

    def _is_yolo(self) -> bool:
        return (
                self._node_type == dai.node.YoloDetectionNetwork or
                self._node_type == dai.node.YoloSpatialDetectionNetwork
        )

    def _is_mobile_net(self) -> bool:
        return (
                self._node_type == dai.node.MobileNetDetectionNetwork or
                self._node_type == dai.node.MobileNetSpatialDetectionNetwork
        )

    def _is_detector(self) -> bool:
        """
        Currently these 2 object detectors are supported
        """
        return self._is_yolo() or self._is_mobile_net()

    def _is_multi_stage(self):
        if not isinstance(self._input, type(self)):
            return False

        if not isinstance(self._input, Component):
            return False

        # if not self._input._is_detector():
        #     raise Exception('Only object detector models can be used as an input to the NNComponent!')

        return True

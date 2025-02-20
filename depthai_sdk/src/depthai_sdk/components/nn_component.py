import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Union, List, Dict

from depthai_sdk.types import NNNode
from depthai_sdk.visualize.bbox import BoundingBox

try:
    import blobconverter
except ImportError:
    blobconverter = None

from depthai_sdk.classes.nn_config import Config
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.integrations.roboflow import RoboflowIntegration
from depthai_sdk.components.multi_stage_nn import MultiStageNN
from depthai_sdk.components.nn_helper import *
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.components.parser import *
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.logger import LOGGER
from depthai_sdk.visualize.visualizer_helper import depth_to_disp_factor
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout, XoutBase
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_nn import XoutTwoStage, XoutNnResults, XoutSpatialBbMappings, XoutNnData
from depthai_sdk.oak_outputs.xout.xout_tracker import XoutTracker
from depthai_sdk.replay import Replay


class NNComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 model: Union[str, Path, Dict],  # str for SDK supported model or Path to custom model's json
                 input: Union[CameraComponent, 'NNComponent', dai.Node.Output],
                 nn_type: Optional[str] = None,  # Either 'yolo' or 'mobilenet'
                 decode_fn: Optional[Callable] = None,
                 tracker: bool = False,  # Enable object tracker - only for Object detection models
                 spatial: Optional[StereoComponent] = None,
                 replay: Optional[Replay] = None,
                 args: Dict = None,  # User defined args
                 ) -> None:
        """
        Neural Network component abstracts:
         - DepthAI API nodes: NeuralNetwork, *DetectionNetwork, *SpatialDetectionNetwork, ObjectTracker
         - Downloading NN models (supported SDK NNs), parsing NN json configs and setting up the pipeline based on it
         - Decoding NN results
         - MultiStage pipelines - cropping high-res frames based on detections and use them for second NN inferencing

        Args:
            model (Union[str, Path, Dict]): str for SDK supported model / Path to blob or custom model's json
            input: (Union[Component, dai.Node.Output, dai.Node.Output]): Input to the NN. If nn_component that is object detector, crop HQ frame at detections (Script node + ImageManip node)
            nn_type (str, optional): Type of the NN - Either 'Yolo' or 'MobileNet'
            tracker (bool, default False): Enable object tracker - only for Object detection models
            spatial (bool, default False): Enable getting Spatial coordinates (XYZ), only for Obj detectors. Yolo/SSD use on-device spatial calc, others on-host (gen2-calc-spatials-on-host)
            replay (Replay object): Replay
            args (Any, optional): Use user defined arguments when constructing the pipeline
        """
        super().__init__()

        self.out = self.Out(self)

        self.triggers = defaultdict(list)
        self.node: Optional[NNNode] = None

        # ImageManip used to resize the input to match the expected NN input size
        self.image_manip: Optional[dai.node.ImageManip] = None
        self.x_in: Optional[dai.node.XLinkIn] = None  # Used for multi-stage pipeline
        # Tracker:
        self.tracker = pipeline.createObjectTracker() if tracker else None
        self.apply_tracking_filter = True  # Enable by default
        self.calculate_speed = True
        self.forget_after_n_frames = None

        # Private properties
        self._ar_resize_mode: ResizeMode = ResizeMode.LETTERBOX  # Default
        # Input to the NNComponent node passed on initialization
        self._input: Union[CameraComponent, 'NNComponent', dai.Node.Output] = input
        self._stream_input: dai.Node.Output  # Node Output that will be used as the input for this NNComponent

        self._blob: Optional[dai.OpenVINO.Blob] = None
        self._forced_version: Optional[dai.OpenVINO.Version] = None  # Forced OpenVINO version
        self._size: Optional[Tuple[int, int]] = None  # Input size to the NN
        self._args: Optional[Dict] = args
        self._config: Optional[Dict] = None
        self._node_type: dai.node = dai.node.NeuralNetwork  # Type of the node for `node`
        self._roboflow: Optional[RoboflowIntegration] = None

        # Multi-stage pipeline
        self._multi_stage_nn: Optional[MultiStageNN] = None

        self._input_queue = Optional[None]  # Input queue for multi-stage pipeline

        self._spatial: Optional[StereoComponent] = spatial
        self._replay: Optional[Replay] = replay  # Replay module

        # For visualizer
        self._labels: Optional[List] = None  # Obj detector labels
        self._handler: Optional[Callable] = None  # Custom model handler for decoding

        # Save passed settings
        self._decode_fn = decode_fn or None  # Decode function that will be used to decode NN results

        # Parse passed settings
        self._parse_model(model)
        if nn_type:
            self._parse_node_type(nn_type)

        # Create NN node
        self.node = pipeline.create(self._node_type)
        self._update_config()

        if self._roboflow:
            path = self._roboflow.device_update(device)
            self._parse_config(path)
            self._update_config()

        if self._blob is None:
            self._blob = dai.OpenVINO.Blob(self._blob_from_config(
                self._config['model'],
                self._config.get('openvino_version', None)
            ))

        # TODO: update NN input based on camera resolution
        self.node.setBlob(self._blob)
        self._out = self.node.out

        if 1 < len(self._blob.networkInputs):
            raise NotImplementedError()

        nn_in: dai.TensorInfo = next(iter(self._blob.networkInputs.values()))
        # TODO: support models that expect mono img
        self._size: Tuple[int, int] = (nn_in.dims[0], nn_in.dims[1])

        # Creates ImageManip node that resizes the input to match the expected NN input size.
        # DepthAI uses CHW (Planar) channel layout and BGR color order convention.
        self.image_manip = pipeline.createImageManip()
        # Configures ImageManip node. Letterbox by default
        self._change_resize_mode(ResizeMode.LETTERBOX)

        if isinstance(self._input, CameraComponent):
            self._stream_input = self._input.stream
            self._stream_input.link(self.image_manip.inputImage)
            # Link ImageManip output to NN node
            self.image_manip.out.link(self.node.input)
        elif isinstance(self._input, dai.Node.Output):
            self._stream_input = self._input
            self._stream_input.link(self.image_manip.inputImage)
            # Link ImageManip output to NN node
            self.image_manip.out.link(self.node.input)
        elif self.is_multi_stage():
            # Here, ImageManip will only crop the high-res frame to correct aspect ratio
            # (without resizing!) and it also acts as a buffer (by default, its pool size is set to 20).
            self.image_manip = pipeline.createImageManip()
            self.image_manip.setNumFramesPool(20)
            self._input._stream_input.link(self.image_manip.inputImage)
            frame_full_size = self._get_input_frame_size()

            if self._input.is_detector():
                self.image_manip.setMaxOutputFrameSize(frame_full_size[0] * frame_full_size[1] * 3)

                # Create script node, get HQ frames from input.
                self._multi_stage_nn = MultiStageNN(pipeline=pipeline,
                                                    detection_node=self._input.node,
                                                    high_res_frames=self.image_manip.out,
                                                    size=self._size,
                                                    frame_size=frame_full_size,
                                                    det_nn_size=self._input._size,
                                                    resize_mode=self._input._ar_resize_mode,
                                                    num_frames_pool=20)
                self._multi_stage_nn.out.link(self.node.input)  # Cropped frames

                # For debugging, for integral counter
                self.node.out.link(self._multi_stage_nn.script.inputs['recognition'])
                self.node.input.setBlocking(True)
                self.node.input.setQueueSize(20)
            else:
                LOGGER.debug('Using on-host decoding for multi-stage NN')
                # Custom NN
                self.image_manip.setResize(*self._size)
                self.image_manip.setMaxOutputFrameSize(self._size[0] * self._size[1] * 3)

                # TODO pass frame on device, and just send config from host
                self.x_in = pipeline.createXLinkIn()
                self.x_in.setStreamName("input_queue")
                self.x_in.setMaxDataSize(frame_full_size[0] * frame_full_size[1] * 3)
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
            self._stereo_node: dai.node.StereoDepth = self._spatial.node
            self._spatial.depth.link(self.node.inputDepth)
            self._spatial.config_stereo(align=self._input)
            # Configure Spatial Detection Network

        if self._args:
            if self.is_spatial():
                self._config_spatials_args(self._args)

    def get_name(self):
        model = self._config.get('model', None)
        if model is not None:
            return model.get('model_name', None)
        return None

    def get_labels(self):
        return [l.upper() if isinstance(l, str) else l[0].upper() for l in self._labels]

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
            
            try:
                zoo_models = blobconverter.zoo_list()
            except Exception as e:
                LOGGER.warning("No internet access, can't load models from depthai zoo.")

            if str(model) in models:
                model = models[str(model)] / 'config.json'
                self._parse_config(model)
            elif str(model) in zoo_models:
                LOGGER.warning(
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
                self._node_type = dai.node.YoloSpatialDetectionNetwork if self.is_spatial() else dai.node.YoloDetectionNetwork
            elif nn_type.upper() == 'MOBILENET':
                self._node_type = dai.node.MobileNetSpatialDetectionNetwork if self.is_spatial() else dai.node.MobileNetDetectionNetwork

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
                from depthai_sdk.integrations.roboflow import RoboflowIntegration
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
                LOGGER.debug("Custom model handler does not contain 'decode' method!")
            else:
                self._decode_fn = self._handler.decode if self._decode_fn is None else self._decode_fn

        if 'nn_config' in self._config:
            nn_config = self._config.get("nn_config", {})

            # Parse node type
            nn_family = nn_config.get("NN_family", None)
            if nn_family:
                self._parse_node_type(nn_family)

    def _blob_from_config(self, model: Dict, version: Union[None, str, dai.OpenVINO.Version] = None) -> str:
        """
        Gets the blob from the config file.
        """
        if isinstance(version, dai.OpenVINO.Version):
            version = str(version)
        if isinstance(version, str):
            if version.startswith('VERSION_'):
                version = version[8:]
            if '_' in version:
                vals = version.split('_')
                version = f'{vals[0]}.{vals[1]}'

        if 'model_name' in model:  # Use blobconverter to download the model
            zoo_type = model.get("zoo", 'intel')
            return blobconverter.from_zoo(model['model_name'],
                                          zoo_type=zoo_type,
                                          shaves=6,  # TODO: Calculate ideal shave amount
                                          version=version)

        if 'xml' in model and 'bin' in model:
            return blobconverter.from_openvino(xml=model['xml'],
                                               bin=model['bin'],
                                               data_type="FP16",  # Myriad X
                                               shaves=6,  # TODO: Calculate ideal shave amount
                                               version=version)

        raise ValueError("Specified `model` values in json config files are incorrect!")

    def _change_resize_mode(self, mode: ResizeMode) -> None:
        """
        Changes the resize mode of the ImageManip node.

        Args:
            mode (ResizeMode): Resize mode to use
        """
        if self.is_multi_stage():
            return  # We need high-res frames for multi-stage NN, so we can crop them later

        self._ar_resize_mode = mode

        # Reset ImageManip node config
        self.image_manip.initialConfig.set(dai.RawImageManipConfig())
        self.image_manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
        self.image_manip.setMaxOutputFrameSize(self._size[0] * self._size[1] * 3)
        self.image_manip.inputImage.setBlocking(False)
        self.image_manip.inputImage.setQueueSize(2)

        if self._ar_resize_mode == ResizeMode.CROP:
            self.image_manip.initialConfig.setResize(self._size)
            # With .setCenterCrop(1), ImageManip will first crop the frame to the correct aspect ratio,
            # and then resize it to the NN input size
            self.image_manip.initialConfig.setCenterCrop(1, self._size[0] / self._size[1])
        elif self._ar_resize_mode == ResizeMode.LETTERBOX:
            self.image_manip.initialConfig.setResizeThumbnail(*self._size)
        elif self._ar_resize_mode == ResizeMode.STRETCH:
            # Not keeping aspect ratio -> stretching the image
            self.image_manip.initialConfig.setKeepAspectRatio(False)
            self.image_manip.initialConfig.setResize(self._size)

    def config_multistage_nn(self,
                             debug=False,
                             labels: Optional[List[int]] = None,
                             scale_bb: Optional[Tuple[int, int]] = None,
                             num_frame_pool: int = None
                             ) -> None:
        """
        Configures the MultiStage NN pipeline. Available if the input to this NNComponent is Detection NNComponent.

        Args:
            debug (bool, default False): Debug script node
            labels (List[int], optional): Crop & run inference only on objects with these labels
            scale_bb (Tuple[int, int], optional): Scale detection bounding boxes (x, y) before cropping the frame. In %.
            num_frame_pool (int, optional): Number of frames to pool for inference. If None, will use the default value.
        """
        if not self.is_multi_stage():
            LOGGER.warning("Input to this model was not a NNComponent, so 2-stage NN inferencing isn't possible!"
                            "This configuration attempt will be ignored.")
            return

        if num_frame_pool is not None:
            # This ImageManip crops the the saved frame based on the detection.
            # Script node sends frame + config to it dynamically.
            self._multi_stage_nn.manip.setNumFramesPool(num_frame_pool)
            # This ImageManip crops the high-res frames and acts as a before
            # prior to Script node (which saves these cropped frames in an array)
            self.image_manip.setNumFramesPool(num_frame_pool)

        self._multi_stage_nn.configure(debug, labels, scale_bb)

    def _parse_label(self, label: Union[str, int]) -> int:
        if isinstance(label, int):
            return label
        elif isinstance(label, str):
            if not self._labels:
                raise ValueError("Incorrect trackLabels type! Make sure to pass NN configuration to "
                                 "the NNComponent so it can deccode string labels!")
            # Label map is Dict of either "name", or ["name", "color"]
            label_strs = [l.upper() if isinstance(l, str) else l[0].upper() for l in self._labels]

            if label.upper() not in label_strs: raise ValueError(f"String '{label}' wasn't found in passed labels!")
            return label_strs.index(label.upper())
        else:
            raise Exception('_parse_label only accepts int or str')

    def config_tracker(self,
                       tracker_type: Optional[dai.TrackerType] = None,
                       track_labels: Optional[List[Union[int, str]]] = None,
                       assignment_policy: Optional[dai.TrackerIdAssignmentPolicy] = None,
                       max_obj: Optional[int] = None,
                       threshold: Optional[float] = None,
                       apply_tracking_filter: Optional[bool] = None,
                       forget_after_n_frames: Optional[int] = None,
                       calculate_speed: Optional[bool] = None
                       ) -> None:
        """
        Configure Object Tracker node (if it's enabled).

        Args:
            tracker_type (dai.TrackerType, optional): Set object tracker type
            track_labels (List[int], optional): Set detection labels to track
            assignment_policy (dai.TrackerType, optional): Set object tracker ID assignment policy
            max_obj (int, optional): Set max objects to track. Max 60.
            threshold (float, optional): Specify tracker threshold. Default: 0.0
            apply_tracking_filter (bool, optional): Set whether to apply Kalman filter to the tracked objects. Done on the host.
            forget_after_n_frames (int, optional): Set how many frames to track an object before forgetting it.
            calculate_speed (bool, optional): Set whether to calculate object speed. Done on the host.
        """

        if self.tracker is None:
            warnings.warn("Tracker was not enabled! Enable with cam.create_nn('[model]', tracker=True)."
                          "This configuration attempt will be ignored.")
            return

        if tracker_type is not None:
            self.tracker.setTrackerType(type=tracker_type)

        if track_labels is not None and 0 < len(track_labels):
            labels = [self._parse_label(l) for l in track_labels]
            self.tracker.setDetectionLabelsToTrack(labels)

        if assignment_policy is not None:
            self.tracker.setTrackerIdAssignmentPolicy(assignment_policy)

        if max_obj is not None:
            if 60 < max_obj:
                raise ValueError("Maximum objects to track is 60!")
            self.tracker.setMaxObjectsToTrack(max_obj)

        if threshold is not None:
            self.tracker.setTrackerThreshold(threshold)

        if apply_tracking_filter is not None:
            self.apply_tracking_filter = apply_tracking_filter

        if forget_after_n_frames is not None:
            self.forget_after_n_frames = forget_after_n_frames

        if calculate_speed is not None:
            self.calculate_speed = calculate_speed

    def config_yolo_from_metadata(self, metadata: Dict) -> None:
        """
        Configures (Spatial) Yolo Detection Network node with a dictionary. Calls config_yolo().
        """
        self.config_yolo(
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
        if not self.is_yolo():
            LOGGER.warning('This is not a YOLO detection network! This configuration attempt will be ignored.')
            return

        if not self.node:
            raise Exception('YOLO node not initialized!')

        self.node.setNumClasses(num_classes)
        self.node.setCoordinateSize(coordinate_size)
        self.node.setAnchors(anchors)
        self.node.setAnchorMasks(masks)
        self.node.setIouThreshold(iou_threshold)

        if conf_threshold is not None:
            self.node.setConfidenceThreshold(conf_threshold)

    def config_nn(self,
                  conf_threshold: Optional[float] = None,
                  resize_mode: Union[ResizeMode, str] = None
                  ) -> None:
        """
        Configures the Detection Network node.

        Args:
            conf_threshold: (float, optional): Confidence threshold for the detections (0..1]
            resize_mode: (ResizeMode, optional): Change aspect ratio resizing mode - to either STRETCH, CROP, or LETTERBOX.
        """
        if resize_mode:
            self._ar_resize_mode = ResizeMode.parse(resize_mode)
            self._change_resize_mode(self._ar_resize_mode)

        if conf_threshold is not None and self.is_detector():
            if 0 <= conf_threshold <= 1:
                self.node.setConfidenceThreshold(conf_threshold)
            else:
                raise ValueError("Confidence threshold must be between 0 and 1!")

    def config_spatial(self,
                       bb_scale_factor: Optional[float] = None,
                       lower_threshold: Optional[int] = None,
                       upper_threshold: Optional[int] = None,
                       calc_algo: Optional[dai.SpatialLocationCalculatorAlgorithm] = None
                       ) -> None:
        """
        Configures the Spatial Detection Network node.

        Args:
            bb_scale_factor (float, optional): Specifies scale factor for detected bounding boxes (0..1]
            lower_threshold (int, optional): Specifies lower threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            upper_threshold (int, optional): Specifies upper threshold in depth units (millimeter by default) for depth values which will used to calculate spatial data
            calc_algo (dai.SpatialLocationCalculatorAlgorithm, optional): Specifies spatial location calculator algorithm: Average/Min/Max
        """
        if not self.is_spatial():
            LOGGER.warning('This is not a Spatial Detection network! This configuration attempt will be ignored.')
            return

        if bb_scale_factor is not None:
            self.node.setBoundingBoxScaleFactor(bb_scale_factor)
        if lower_threshold is not None:
            self.node.setDepthLowerThreshold(lower_threshold)
        if upper_threshold is not None:
            self.node.setDepthUpperThreshold(upper_threshold)
        if calc_algo:
            self.node.setSpatialCalculationAlgorithm(calc_algo)

    def _update_config(self) -> None:
        if self.node is None or self._config is None:
            return

        nn_config = self._config.get("nn_config", {})

        meta = nn_config.get('NN_specific_metadata', None)
        if self.is_yolo() and meta:
            self.config_yolo_from_metadata(metadata=meta)

        self.config_nn(conf_threshold=nn_config.get('conf_threshold', None))

    def _get_camera_comp(self) -> CameraComponent:
        if self.is_multi_stage():
            return self._input._get_camera_comp()
        return self._input

    def _get_input_frame_size(self) -> Tuple[int, int]:
        # TODO: if user passes node output as the NN input (eg. examples/mixed/switch_between_models.py),
        # this function will fail
        return self._get_camera_comp().stream_size

    #
    def get_bbox(self) -> BoundingBox:
        if self.is_multi_stage():
            return self._input.get_bbox()
        else:
            try:
                stream_size = self._get_input_frame_size()
                old_ar = stream_size[0] / stream_size[1]
                new_ar = self._size[0] / self._size[1]
                return BoundingBox().resize_to_aspect_ratio(old_ar, new_ar, self._ar_resize_mode)
            except (AttributeError, ZeroDivisionError, ValueError):
                return BoundingBox()

    """
    Available outputs (to the host) of this component
    """

    class Out:

        class MainOut(ComponentOutput):
            """
            Default output. Streams NN results and high-res frames that were downscaled and used for inferencing.
            Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
            """

            def __call__(self, device: dai.Device, fourcc: Optional[str] = None) -> XoutBase:
                if self._comp.is_multi_stage():
                    det_nn_out = StreamXout(out=self._comp._input.node.out)
                    second_nn_out = StreamXout(out=self._comp.node.out)

                    return XoutTwoStage(det_nn=self._comp._input,
                                        second_nn=self._comp,
                                        frames=self._comp._input._input.get_stream_xout(),
                                        det_out=det_nn_out,
                                        second_nn_out=second_nn_out,
                                        device=device,
                                        input_queue_name="input_queue" if self._comp.x_in else None,
                                        bbox=self._comp.get_bbox()).set_fourcc(fourcc).set_comp_out(self)
                else:
                    # TODO: refactor. This is a bit hacky, as we want to support passing node output as the input
                    # to the NNComponent. In such case, we don't have access to VideoEnc (inside CameraComponent)
                    det_nn_out = StreamXout(out=self._comp.node.out)
                    input_stream = self._comp._stream_input
                    if fourcc is None:
                        frame_stream = StreamXout(out=input_stream)
                    else:
                        frame_stream = self._comp._get_camera_comp().get_stream_xout(fourcc)
                    return XoutNnResults(det_nn=self._comp,
                                         frames=frame_stream,
                                         nn_results=det_nn_out,
                                         bbox=self._comp.get_bbox()).set_fourcc(fourcc).set_comp_out(self)

        class PassThroughOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                """
                Default output. Streams NN results and passthrough frames (frames used for inferencing)
                Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
                """
                if self._comp.is_multi_stage():
                    return XoutTwoStage(det_nn=self._comp._input,
                                        second_nn=self._comp,
                                        frames=StreamXout(out=self._comp._input.node.passthrough),
                                        det_out=StreamXout(out=self._comp._input.node.out),
                                        second_nn_out=StreamXout(self._comp.node.out),
                                        device=device,
                                        input_queue_name="input_queue" if self._comp.x_in else None,
                                        bbox=self._comp.get_bbox()).set_comp_out(self)
                else:
                    return XoutNnResults(det_nn=self._comp,
                                         frames=StreamXout(out=self._comp.node.passthrough),
                                         nn_results=StreamXout(out=self._comp.node.out),
                                         bbox=BoundingBox()
                                         ).set_comp_out(self)

        class ImgManipOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                return XoutFrames(StreamXout(out=self._comp.image_manip.out)).set_comp_out(self)

        class InputOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutBase:
                return XoutFrames(StreamXout(out=self._comp._stream_input)).set_comp_out(self)

        class SpatialOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutSpatialBbMappings:
                """
                Streams depth and bounding box mappings (``SpatialDetectionNework.boundingBoxMapping``). Produces SpatialBbMappingPacket.
                """
                if not self._comp.is_spatial():
                    raise Exception('SDK tried to output spatial data (depth + bounding box mappings),'
                                    'but this is not a Spatial Detection network!')

                return XoutSpatialBbMappings(
                    device=device,
                    stereo=self._comp._stereo_node,
                    frames=StreamXout(out=self._comp.node.passthroughDepth),
                    configs=StreamXout(out=self._comp.node.out),
                    dispScaleFactor=depth_to_disp_factor(device, self._comp._stereo_node),
                    bbox=self._comp.get_bbox()
                ).set_comp_out(self)

        class TwoStageOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutFrames:
                """
                Streams 2. stage cropped frames to the host. Produces FramePacket.
                """
                if not self._comp.is_multi_stage():
                    raise Exception(
                        'SDK tried to output TwoStage crop frames, but this is not a Two-Stage NN component!')

                return XoutFrames(frames=StreamXout(out=self._comp._multi_stage_nn.manip.out)).set_comp_out(self)

        class TrackerOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutTracker:
                """
                Streams ObjectTracker tracklets and high-res frames that were downscaled and used for inferencing. Produces TrackerPacket.
                """
                if not self._comp.is_tracker():
                    raise Exception('Tracker was not enabled! Enable with cam.create_nn("[model]", tracker=True)!')

                self._comp.node.passthrough.link(self._comp.tracker.inputDetectionFrame)
                self._comp.node.out.link(self._comp.tracker.inputDetections)

                # TODO: add support for full frame tracking
                self._comp.node.passthrough.link(self._comp.tracker.inputTrackerFrame)

                return XoutTracker(det_nn=self._comp,
                                   frames=self._comp._input.get_stream_xout(),  # CameraComponent
                                   device=device,
                                   tracklets=StreamXout(self._comp.tracker.out),
                                   bbox=self._comp.get_bbox(),
                                   apply_kalman=self._comp.apply_tracking_filter,
                                   forget_after_n_frames=self._comp.forget_after_n_frames,
                                   calculate_speed=self._comp.calculate_speed,
                                   ).set_comp_out(self)

        class EncodedOut(MainOut):
            def __call__(self, device: dai.Device) -> XoutNnResults:
                """
                Streams NN results and encoded frames (frames used for inferencing)
                Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
                """
                # A bit hacky, maybe we can remove this alltogether
                return super().__call__(device, fourcc=self._comp._get_camera_comp().get_fourcc())

        class NnDataOut(ComponentOutput):
            def __call__(self, device: dai.Device) -> XoutNnData:
                node_output = self._comp.node.out if \
                    type(self._comp.node) == dai.node.NeuralNetwork else \
                    self._comp.node.outNetwork

                return XoutNnData(xout=StreamXout(node_output)).set_comp_out(self)

        def __init__(self, nn_component: 'NNComponent'):
            self.main = self.MainOut(nn_component)
            self.passthrough = self.PassThroughOut(nn_component)
            self.image_manip = self.ImgManipOut(nn_component)
            self.input = self.InputOut(nn_component)
            self.spatials = self.SpatialOut(nn_component)
            self.twostage_crops = self.TwoStageOut(nn_component)
            self.tracker = self.TrackerOut(nn_component)
            self.encoded = self.EncodedOut(nn_component)
            self.nn_data = self.NnDataOut(nn_component)

    # Checks
    def is_spatial(self) -> bool:
        return self._spatial is not None  # todo fix if spatial is bool and equals to False

    def is_tracker(self) -> bool:
        # Currently, only object detectors are supported
        return self.is_detector() and self.tracker is not None

    def is_yolo(self) -> bool:
        return (
                self._node_type == dai.node.YoloDetectionNetwork or
                self._node_type == dai.node.YoloSpatialDetectionNetwork
        )

    def is_mobile_net(self) -> bool:
        return (
                self._node_type == dai.node.MobileNetDetectionNetwork or
                self._node_type == dai.node.MobileNetSpatialDetectionNetwork
        )

    def is_detector(self) -> bool:
        """
        Currently these 2 object detectors are supported
        """
        return self.is_yolo() or self.is_mobile_net()

    def is_multi_stage(self):
        if not isinstance(self._input, type(self)):
            return False

        if not isinstance(self._input, Component):
            return False

        # if not self._input._is_detector():
        #     raise Exception('Only object detector models can be used as an input to the NNComponent!')

        return True

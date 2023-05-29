import numpy as np
import cv2
from pathlib import Path
import marshal
from string import Template
from math import sin, cos
import depthai_sdk.components.pose_estimation.mediapipe_utils as mpu
from typing import Union, Dict, Optional

import depthai as dai
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout, XoutBase
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_pose_estimation import XoutPoseDetections, XoutBlazepose, XoutBlazeposePassthrough

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = str(SCRIPT_DIR / "models/pose_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/pose_landmark_full_sh4.blob")
LANDMARK_MODEL_HEAVY = str(SCRIPT_DIR / "models/pose_landmark_heavy_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(
    SCRIPT_DIR / "custom_models/DetectionBestCandidate_sh1.blob")
DIVIDE_BY_255_MODEL = str(SCRIPT_DIR / "custom_models/DivideBy255_sh1.blob")


class BlazeposeComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 input: CameraComponent,
                 #  model: str = 'lite',  # "lite", "full", or "sparse"
                 # If passed, enable XYZ
                 spatial: Union[None, StereoComponent] = None,
                 args: Dict = None,  # User defined args
                 name: Optional[str] = None
                 ) -> None:
        """
        This component aims to port https://github.com/geaxgx/depthai_hand_tracker into the DepthAI SDK. It uses Google Mediapipe's hand landmark model.
        """
        super().__init__()

        self.name = name
        self.out = self.Out(self)

        self.pd_model = POSE_DETECTION_MODEL
        self.pp_model = DETECTION_POSTPROCESSING_MODEL
        self.divide_by_255_model = DIVIDE_BY_255_MODEL
        self.lm_model = LANDMARK_MODEL_FULL

        self._input: CameraComponent = input

        self.rect_transf_scale = 1.25

        self.pd_score_thresh = 0.5
        self.lm_score_thresh = 0.7
        self.smoothing = True
        self.crop = False  # Letterbox?
        self.internal_fps = None
        self.presence_threshold = 0.5
        self.visibility_threshold = 0.5
        self.trace = False
        self.force_detection = False

        self.xyz = spatial is not None

        self.img_w, self.img_h = self._input.stream_size
        self.crop_w = 0
        self.frame_size = self.img_w
        self.pad_w = 0
        self.pad_h = (self.img_w - self.img_h) // 2

        self.resize_mode = ResizeMode.LETTERBOX

        self.nb_kps = 33

        if self.smoothing:
            self.video_fps = 20
            self.filter_landmarks = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.05,
                beta=80,
                derivate_cutoff=1
            )
            # landmarks_aux corresponds to the 2 landmarks used to compute the ROI in next frame
            self.filter_landmarks_aux = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.01,
                beta=10,
                derivate_cutoff=1
            )
            self.filter_landmarks_world = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.1,
                beta=40,
                derivate_cutoff=1,
                disable_value_scaling=True
            )
            if self.xyz:
                self.filter_xyz = mpu.LowPassFilter(alpha=0.25)

        self.pd_input_length = 224
        self.lm_input_length = 256

        # Define manager script node
        self.script = pipeline.create(dai.node.Script)
        self.script.setScript(self.build_manager_script())
        self.script.setProcessor(dai.ProcessorType.LEON_CSS)  # More stable

        if self.xyz:
            slc = pipeline.createSpatialLocationCalculator()
            slc.setWaitForConfigInput(True)
            slc.inputDepth.setBlocking(False)
            slc.inputDepth.setQueueSize(1)

            spatial.depth.link(slc.inputDepth)

            self.script.outputs['spatial_location_config'].link(slc.inputConfig)
            slc.out.link(self.script.inputs['spatial_data'])

        # Define pose detection pre processing (resize preview to (self.pd_input_length, self.pd_input_length))
        self.pre_pd_manip = pipeline.create(dai.node.ImageManip)
        self.pre_pd_manip.setMaxOutputFrameSize(
            self.pd_input_length*self.pd_input_length*3)
        self.pre_pd_manip.setWaitForConfigInput(True)
        self.pre_pd_manip.inputImage.setQueueSize(1)
        self.pre_pd_manip.inputImage.setBlocking(False)
        self._input.stream.link(self.pre_pd_manip.inputImage)
        self.script.outputs['pre_pd_manip_cfg'].link(
            self.pre_pd_manip.inputConfig)

        # Define pose detection model
        self.pd_nn = pipeline.create(dai.node.NeuralNetwork)
        self.pd_nn.setBlobPath(self.pd_model)
        self.pre_pd_manip.out.link(self.pd_nn.input)

        # Define pose detection post processing "model"
        self.post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        self.post_pd_nn.setBlobPath(self.pp_model)
        self.pd_nn.out.link(self.post_pd_nn.input)
        self.post_pd_nn.out.link(self.script.inputs['from_post_pd_nn'])

        # Define landmark pre processing image manip
        self.pre_lm_manip = pipeline.create(dai.node.ImageManip)
        self.pre_lm_manip.setMaxOutputFrameSize(
            self.lm_input_length*self.lm_input_length*3)
        self.pre_lm_manip.setWaitForConfigInput(True)
        self.pre_lm_manip.inputImage.setQueueSize(1)
        self.pre_lm_manip.inputImage.setBlocking(False)
        self._input.stream.link(self.pre_lm_manip.inputImage)

        self.script.outputs['pre_lm_manip_cfg'].link(self.pre_lm_manip.inputConfig)

        # Define normalization model between ImageManip and landmark model
        # This is a temporary step. Could be removed when support of setFrameType(RGBF16F16F16p) in ImageManip node
        divide_nn = pipeline.create(dai.node.NeuralNetwork)
        divide_nn.setBlobPath(self.divide_by_255_model)
        self.pre_lm_manip.out.link(divide_nn.input)

        # Define landmark model
        self.lm_nn = pipeline.create(dai.node.NeuralNetwork)
        self.lm_nn.setBlobPath(self.lm_model)

        divide_nn.out.link(self.lm_nn.input)
        self.lm_nn.out.link(self.script.inputs['from_lm_nn'])

    def build_manager_script(self):
        '''
        The code of the scripting node 'self.script' depends on :
            - the NN model (full, lite, 831),
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_self.script.py which is a python template
        '''
        path = Path(__file__).resolve().parent / "template_manager_script.py"
        with open(path, 'r', encoding='utf-8') as file:
            template = Template(file.read())

        # Perform the substitution
        code = template.substitute(
            _TRACE="node.warn" if self.trace else "#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=self.lm_score_thresh,
            _force_detection=self.force_detection,
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w,
            _frame_size=self.frame_size,
            _crop_w=self.crop_w,
            _rect_transf_scale=self.rect_transf_scale,
            _IF_XYZ="" if self.xyz else '"""',
            _buffer_size=2910 if self.xyz else 2863,
            _visibility_threshold=self.visibility_threshold
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace:
            with open("tmp_code.py", "w") as file:
                file.write(code)

        return code


    class Out:
        """
        Available outputs (to the host) of this component
        """

        def __init__(self, nn_component: 'HandTrackerComponent'):
            self._comp = nn_component

        def main(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Default output. Streams NN results and high-res frames that were downscaled and used for inferencing.
            """
            script_out = StreamXout(out=self._comp.script.outputs['host'],
                                    name='host')
            out = XoutBlazepose(
                script_stream=script_out,
                frames=StreamXout(out=self._comp._input.stream),
                component=self._comp
            )
            return self._comp._create_xout(pipeline, out)

        def passthrough(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            script_out = StreamXout(out=self._comp.script.outputs['host'], name='host2')
            out = XoutBlazeposePassthrough(
                script_stream=script_out,
                frames=StreamXout(out=self._comp.pre_lm_manip.out),
                component=self._comp
            )
            return self._comp._create_xout(pipeline, out)

        def pose_detection(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Only pose detection outputs. Send both PD decoded results + passthrough frames (frames
            used for inferencing PD)
            """
            out = XoutPoseDetections(
                frames=StreamXout(out=self._comp.pd_nn.passthrough),
                nn_results=StreamXout(out=self._comp.post_pd_nn.out, name="Pose detection results"),
                component=self._comp
            )
            return self._comp._create_xout(pipeline, out)

        def pose_crop(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Send out the cropped Pose frames, which are sent to the pose landmark NN. Useful for debugging.
            """
            out = XoutFrames(frames=StreamXout(out=self._comp.pre_lm_manip.out))
            return self._comp._create_xout(pipeline, out)

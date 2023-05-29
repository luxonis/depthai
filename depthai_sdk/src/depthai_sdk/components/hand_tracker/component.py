from pathlib import Path
from typing import  Union, Dict, Optional
from string import Template

from depthai_sdk.oak_outputs.xout.xout_hand_tracker import XoutHandTracker, XoutPalmDetection

import blobconverter
import depthai as dai
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout, XoutBase
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames

class HandTrackerComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline,
                 input: CameraComponent,
                #  model: str = 'lite',  # "lite", "full", or "sparse"
                 spatial: Union[None, StereoComponent] = None, # If passed, enable XYZ
                 args: Dict = None,  # User defined args
                 name: Optional[str] = None
                 ) -> None:
        """
        This component aims to port https://github.com/geaxgx/depthai_hand_tracker into the DepthAI SDK. It uses Google Mediapipe's hand landmark model.
        """
        super().__init__()

        self.name = name
        self.out = self.Out(self)

        # Private properties
        self._ar_resize_mode: ResizeMode = ResizeMode.LETTERBOX  # Default
        self._input: CameraComponent = input  # Input to the component node passed on initialization
        self._stream_input: dai.Node.Output  # Node Output that will be used as the input for this NNComponent

        self._args: Optional[Dict] = args

        self._spatial: Optional[Union[bool, StereoComponent]] = spatial


        self.use_lm = True
        self.trace = 0
        self.lm_nb_threads = 2 # Number of inference threads for the landmark model
        self.xyz = False if spatial is None else True
        self.crop = False
        self.use_world_landmarks = False
        self.use_gesture = False
        self.use_handedness_average = True
        self.single_hand_tolerance_thresh = 10
        self.use_same_image = True
        self.pd_score_thresh = 0.5
        self.lm_score_thresh = 0.4
        self.pd_resize_mode = ResizeMode.LETTERBOX

        self.img_w, self.img_h = self._input.stream_size

        self.crop_w = 0
        self.frame_size = self.img_w
        self.pad_w = 0
        self.pad_h = (self.img_w - self.img_h) // 2

        self.script = pipeline.create(dai.node.Script)
        self.script.setScript(self.build_manager_script())
        self.script.setProcessor(dai.ProcessorType.LEON_CSS)  # More stable

        if spatial is not None:
            slc = pipeline.create(dai.node.SpatialLocationCalculator)
            slc.setWaitForConfigInput(True)
            slc.inputDepth.setBlocking(False)
            slc.inputDepth.setQueueSize(1)

            spatial.node.depth.link(slc.inputDepth)
            self.script.outputs['spatial_location_config'].link(slc.inputConfig)
            slc.out.link(self.script.inputs['spatial_data'])

        # Palm detection ---------
        self.pd_size = (128, 128)
        pd_manip = pipeline.create(dai.node.ImageManip)
        pd_manip.setMaxOutputFrameSize(self.pd_size[0] * self.pd_size[1] * 3)
        pd_manip.setWaitForConfigInput(True)
        pd_manip.inputImage.setQueueSize(1)
        pd_manip.inputImage.setBlocking(False)

        # if self._ar_resize_mode == ResizeMode.CROP:
        #     pd_manip.initialConfig.setKeepAspectRatio(False) # Stretch the image to fit the NN input
        #     self.image_manip.initialConfig.setCenterCrop(1, self._size[0] / self._size[1])
        # elif self._ar_resize_mode == ResizeMode.LETTERBOX:
        #     pd_manip.initialConfig.setKeepAspectRatio(True)
        # elif self._ar_resize_mode == ResizeMode.STRETCH:
        #     pd_manip.initialConfig.setKeepAspectRatio(False)

        self._input.stream.link(pd_manip.inputImage)
        self.script.outputs['pre_pd_manip_cfg'].link(pd_manip.inputConfig)

        self.palm_detection_nn = pipeline.create(dai.node.NeuralNetwork)
        self.palm_detection_nn.setBlobPath(blobconverter.from_zoo(name="palm_detection_128x128",
                                                 zoo_type='depthai',
                                                 shaves=6))
        pd_manip.out.link(self.palm_detection_nn.input)

        # Palm detection decoding -----------
        self.palm_detection_decoding = pipeline.create(dai.node.NeuralNetwork)
        self.palm_detection_decoding.setBlobPath(blobconverter.from_zoo(name="palm_detection_128x128_decoding",
                                                      zoo_type='depthai',
                                                      shaves=6,
                                                      # Required, otherwise it will try to convert INT8 to FP16:
                                                      compile_params=[]
                                                      ))
        self.palm_detection_nn.out.link(self.palm_detection_decoding.input)
        self.palm_detection_decoding.out.link(self.script.inputs['from_post_pd_nn'])

        # Hand landmarks ------------

        self.lm_input_length = 224
        self.pre_lm_manip = pipeline.create(dai.node.ImageManip)
        self.pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length*self.lm_input_length*3)
        self.pre_lm_manip.setWaitForConfigInput(True)
        self.pre_lm_manip.inputImage.setQueueSize(1)
        self.pre_lm_manip.inputImage.setBlocking(False)
        self._input.stream.link(self.pre_lm_manip.inputImage)

        self.script.outputs['pre_lm_manip_cfg'].link(self.pre_lm_manip.inputConfig)

        # Define landmark model
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(blobconverter.from_zoo(name="hand_landmark_224x224",
                                               zoo_type='depthai',
                                               shaves=6))
        lm_nn.setNumInferenceThreads(self.lm_nb_threads)
        self.pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(self.script.inputs['from_lm_nn'])


    def build_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script_*.py which is a python template
        '''
        path = Path(__file__).resolve().parent / "template_manager_script_duo.py"

        # Read the template
        with open(path, 'r', encoding='utf-8') as file:
            template = Template(file.read())

        # Perform the substitution
        code = template.substitute(
                    _TRACE1 = "node.warn" if self.trace & 1 else "#",
                    _TRACE2 = "node.warn" if self.trace & 2 else "#",
                    _pd_score_thresh = self.pd_score_thresh,
                    _lm_score_thresh = self.lm_score_thresh,
                    _pad_h = self.pad_h,
                    _img_h = self.img_h,
                    _img_w = self.img_w,
                    _frame_size = self.frame_size,
                    _crop_w = self.crop_w,
                    _IF_XYZ = "" if self.xyz else '"""',
                    _IF_USE_HANDEDNESS_AVERAGE = "" if self.use_handedness_average else '"""',
                    _single_hand_tolerance_thresh= self.single_hand_tolerance_thresh,
                    _IF_USE_SAME_IMAGE = "" if self.use_same_image else '"""',
                    _IF_USE_WORLD_LANDMARKS = "" if self.use_world_landmarks else '"""',
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
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
            Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
            """
            # manager_out = pipeline.create(dai.node.XLinkOut)
            # manager_out.setStreamName("manager_out")
            # manager_.link(manager_out.input)
            script_out = StreamXout(out=self._comp.script.outputs['host'],
                                    name='host')

            camera_comp = self._comp._input
            frames = StreamXout(out=camera_comp.stream)

            out = XoutHandTracker(
                script_stream=script_out,
                frames=frames,
                component=self._comp
            )
            return self._comp._create_xout(pipeline, out)

        def palm_detection(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Only palm detection outputs. Send both PD decoded results + passthrough frames (frames
            used for inferencing PD)
            """
            out = XoutPalmDetection(
                frames=StreamXout(out=self._comp.palm_detection_nn.passthrough),
                nn_results=StreamXout(out=self._comp.palm_detection_decoding.out, name="Palm detection results"),
                hand_tracker=self._comp
            )
            return self._comp._create_xout(pipeline, out)

        def palm_crop(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
            """
            Send out the cropped Palm frames, which are sent to the hand landmark NN. Useful for debugging.
            """
            out = XoutFrames(
                frames=StreamXout(out=self._comp.pre_lm_manip.out)
                )
            return self._comp._create_xout(pipeline, out)



        # def passthrough(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        #     """
        #     Default output. Streams NN results and passthrough frames (frames used for inferencing)
        #     Produces DetectionPacket or TwoStagePacket (if it's 2. stage NNComponent).
        #     """
        #     det_nn_out = StreamXout(out=self._comp.node.out, name=self._comp.name)
        #     frames = StreamXout(out=self._comp.node.passthrough, name=self._comp.name)

        #     out = XoutNnResults(det_nn=self._comp,
        #                         frames=frames,
        #                         nn_results=det_nn_out)

        #     return self._comp._create_xout(pipeline, out)


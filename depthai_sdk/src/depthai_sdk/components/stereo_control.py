import depthai as dai
from itertools import cycle
from depthai_sdk.components.parser import parse_median_filter
from depthai_sdk.logger import LOGGER

LIMITS = {
    'confidence_threshold': (0, 255),
    'bilateral_sigma': (0, 255),
    'range': (0, 65535),
    'lrc_threshold': (0, 10),
    'dot_projector': (0, 1200),
    'illumination_led': (0, 1500),
}


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class StereoControl:
    def __init__(self, device: dai.Device):
        self.queue = None

        self._cycle_median_filter = cycle([item for name, item in vars(dai.StereoDepthConfig.MedianFilter).items() if name[0].isupper()])
        self.device = device

        self._current_vals = {
            'conf_threshold': 240,
            'dot_projector': 800,
            'illumination_led': 200,
        }

        ctrl = dai.StereoDepthConfig()
        self.raw_cfg = ctrl.get()

    def set_input_queue(self, queue: dai.DataInputQueue):
        self.queue = queue

    def switch_median_filter(self):
        """
        Switch between auto white balance modes.
        """
        mode = next(self._cycle_median_filter)
        LOGGER.info(f'Switching median filter to {mode}')
        self.send_controls({'postprocessing': {'median': mode}})

    def confidence_threshold_up(self, step=10):
        """
        Increase confidence threshold by step
        """
        if LIMITS['confidence_threshold'][1] < self._current_vals['conf_threshold'] + step:
            LOGGER.error(f'Confidence threshold cannot be greater than {LIMITS["confidence_threshold"][1]}')
            return
        self._current_vals['conf_threshold'] += step
        self.send_controls({'cost_matching': {'confidence_threshold': self._current_vals['conf_threshold']}})

    def confidence_threshold_down(self, step=10):
        """
        Decrease confidence threshold by step
        """
        if LIMITS['confidence_threshold'][0] > self._current_vals['conf_threshold'] - step:
            LOGGER.error(f'Confidence threshold cannot be less than {LIMITS["confidence_threshold"][0]}')
            return
        self._current_vals['conf_threshold'] -= step
        self.send_controls({'cost_matching': {'confidence_threshold': self._current_vals['conf_threshold']}})

    def dot_projector_up(self, step=50):
        """
        Increase dot projector power by step
        """
        if LIMITS['dot_projector'][1] < self._current_vals['dot_projector'] + step:
            LOGGER.error(f'Dot projector power cannot be greater than {LIMITS["dot_projector"][1]}')
            return
        self._current_vals['dot_projector'] += step
        self.device.setIrFloodLightBrightness(self._current_vals['dot_projector'])

    def dot_projector_down(self, step=50):
        """
        Decrease dot projector power by step
        """
        if LIMITS['dot_projector'][0] > self._current_vals['dot_projector'] - step:
            LOGGER.error(f'Dot projector power cannot be less than {LIMITS["dot_projector"][0]}')
            return
        self._current_vals['dot_projector'] -= step
        self.device.setIrFloodLightBrightness(self._current_vals['dot_projector'])

    def illumination_led_up(self, step=50):
        """
        Increase illumination led power by step
        """
        if LIMITS['illumination_led'][1] < self._current_vals['illumination_led'] + step:
            LOGGER.error(f'Illumination led power cannot be greater than {LIMITS["illumination_led"][1]}')
            return
        self._current_vals['illumination_led'] += step
        self.device.setIrLaserDotProjectorBrightness(self._current_vals['illumination_led'])

    def illumination_led_down(self, step=50):
        """
        Decrease illumination led power by step
        """
        if LIMITS['illumination_led'][0] > self._current_vals['illumination_led'] - step:
            LOGGER.error(f'Illumination led power cannot be less than {LIMITS["illumination_led"][0]}')
            return
        self._current_vals['illumination_led'] -= step
        self.device.setIrLaserDotProjectorBrightness(self._current_vals['illumination_led'])

    def send_controls(self, controls: dict):
        """
        Send controls to the StereoDepth node. Dict structure and available options:

        ctrl = {
            'algorithm_control':{
                'align': 'RECTIFIED_RIGHT', # | 'RECTIFIED_LEFT' | 'CENTER'
                'unit': 'METER', # | 'CENTIMETER' | 'MILLIMETER' | 'INCH' | 'FOOT' | 'CUSTOM'
                'unit_multiplier': 1000, # Only if 'unit' is 'CUSTOM'
                'lr_check': True, # Enable left-right check
                'extended': True, # Enable extended disparity
                'subpixel': True, # Enable subpixel disparity
                'lr_check_threshold': 10, # Left-right check threshold
                'subpixel_bits': 3, # 3 | 4 | 5
                'disparity_shift': 0, # Disparity shift
                'invalidate_edge_pixels': 0 # Number of pixels to invalidate at the edge of the image
            },
            'postprocessing': {
                'median': 5, # 0 | 3 | 5 | 7
                'bilateral_sigma': 0, # Sigma value for bilateral filter
                'spatial': {
                    'enable': True, # Enable spatial denoise
                    'hole_filling': 2, # Hole filling radius
                    'alpha': 0.5, # Alpha factor in an exponential moving average
                    'delta': 0, # Step-size boundary
                    'iterations': 1, # Number of iterations
                },
                'temporal': {
                    'enable': False,  # Enable or disable temporal denoise
                    'persistency_mode': 3,  # Persistency mode (use corresponding integer value for the enum member)
                    'alpha': 0.4,  # Alpha factor in an exponential moving average
                    'delta': 0,  # Step-size boundary
                },
                'threshold': {
                    'min_range': 0,  # Minimum range in depth units
                    'max_range': 65535,  # Maximum range in depth units
                },
                'brightness': {
                    'min': 0,  # Minimum pixel brightness
                    'max': 256,  # Maximum pixel brightness
                },
                'speckle': {
                    'enable': False,  # Enable or disable the speckle filter
                    'range': 50,  # Speckle search range
                },
                'decimation': {
                    'factor': 1,  # Decimation factor (1, 2, 3, or 4)
                    'mode': 0,  # Decimation algorithm type (use corresponding integer value for the enum member)
                }
            },
            'census_transform': {
                'kernel_size': 'AUTO',  # | 'KERNEL_5x5' | 'KERNEL_7x7' | 'KERNEL_7x9'
                'kernel_mask': 0,  # Census transform mask
                'enable_mean_mode': True,  # Enable mean mode
                'threshold': 0,  # Census transform comparison threshold value
            },
            'cost_matching': {
                'disparity_width': 'DISPARITY_64',  # or 'DISPARITY_96'
                'enable_companding': False,  # Enable disparity companding using sparse matching
                'confidence_threshold': 245,  # Confidence threshold for accepted disparities
                'linear_equation_parameters': {
                    'alpha': 0,
                    'beta': 2,
                    'threshold': 127,
                },
            },
            'cost_aggregation': {
                'division_factor': 1,  # Division factor for cost calculation linear equation parameters
                'horizontal_penalty_cost_p1': 250,  # Horizontal P1 penalty cost parameter
                'horizontal_penalty_cost_p2': 500,  # Horizontal P2 penalty cost parameter
                'vertical_penalty_cost_p1': 250,  # Vertical P1 penalty cost parameter
                'vertical_penalty_cost_p2': 500,  # Vertical P2 penalty cost parameter
            },
            'reset': False # Reset all controls to default
        }
        """
        if self.queue is None:
            LOGGER.error('Cannot send controls when replaying.')
            return

        LOGGER.info(f'Sending controls to StereoDepth node: {controls}')

        ctrl = dai.StereoDepthConfig()

        if controls.get('reset', None) and controls['reset']:
            LOGGER.info('Resetting camera controls.')
            self.raw_cfg = ctrl.get()
            ctrl.set(self.raw_cfg)
            self.queue.send(ctrl)
            return

        if controls.get('algorithm_control', None) is not None:
            if controls['algorithm_control'].get('align', None) is not None:
                if isinstance(controls["algorithm_control"]["align"], str):
                    controls["algorithm_control"]["align"] = getattr(dai.StereoDepthConfig.AlgorithmControl.DepthAlign, controls["algorithm_control"]["align"])
                self.raw_cfg.algorithmControl.depthAlign = controls["algorithm_control"]["align"]
            if controls['algorithm_control'].get('unit', None) is not None:
                if isinstance(controls["algorithm_control"]["unit"], str):
                    controls["algorithm_control"]["unit"] = getattr(dai.StereoDepthConfig.AlgorithmControl.DepthUnit, controls["algorithm_control"]["unit"])
                self.raw_cfg.algorithmControl.depthUnit = controls["algorithm_control"]["unit"]
            if controls['algorithm_control'].get('unit_multiplier', None) is not None:
                self.raw_cfg.algorithmControl.customDepthUnitMultiplier = controls["algorithm_control"]["unit_multiplier"]
            if controls['algorithm_control'].get('lr_check', None) is not None:
                self.raw_cfg.algorithmControl.enableLeftRightCheck = controls["algorithm_control"]["lr_check"]
            if controls['algorithm_control'].get('extended', None) is not None:
                self.raw_cfg.algorithmControl.enableExtended = controls["algorithm_control"]["extended"]
            if controls['algorithm_control'].get('subpixel', None) is not None:
                self.raw_cfg.algorithmControl.enableSubpixel = controls["algorithm_control"]["subpixel"]
            if controls['algorithm_control'].get('lr_check_threshold', None) is not None:
                lrc_threshold = clamp(controls["algorithm_control"]["lr_check_threshold"], *LIMITS['lrc_threshold'])
                self.raw_cfg.algorithmControl.leftRightCheckThreshold = lrc_threshold
            if controls['algorithm_control'].get('subpixel_bits', None) is not None:
                self.raw_cfg.algorithmControl.subpixelFractionalBits = controls["algorithm_control"]["subpixel_bits"]
            if controls['algorithm_control'].get('disparity_shift', None) is not None:
                self.raw_cfg.algorithmControl.disparityShift = controls["algorithm_control"]["disparity_shift"]
            if controls['algorithm_control'].get('invalidate_edge_pixels', None) is not None:
                self.raw_cfg.algorithmControl.numInvalidateEdgePixels = controls["algorithm_control"]["invalidate_edge_pixels"]

        if controls.get('postprocessing', None) is not None:
            if controls['postprocessing'].get('median', None) is not None:
                self.raw_cfg.postProcessing.median = parse_median_filter(controls["postprocessing"]["median"])
            if controls['postprocessing'].get('bilateral_sigma', None) is not None:
                bilateral_sigma = clamp(controls["postprocessing"]["bilateral_sigma"], *LIMITS['bilateral_sigma'])
                self.raw_cfg.postProcessing.bilateralSigmaValue = bilateral_sigma

            if controls['postprocessing'].get('spatial', None) is not None:
                if controls['postprocessing']['spatial'].get('enable', None) is not None:
                    self.raw_cfg.postProcessing.spatialFilter.enable = controls["postprocessing"]["spatial"]["enable"]
                if controls['postprocessing']['spatial'].get('hole_filling', None) is not None:
                    self.raw_cfg.postProcessing.spatialFilter.holeFillingRadius = controls["postprocessing"]["spatial"]["hole_filling"]
                if controls['postprocessing']['spatial'].get('alpha', None) is not None:
                    self.raw_cfg.postProcessing.spatialFilter.alpha = controls["postprocessing"]["spatial"]["alpha"]
                if controls['postprocessing']['spatial'].get('delta', None) is not None:
                    self.raw_cfg.postProcessing.spatialFilter.delta = controls["postprocessing"]["spatial"]["delta"]
                if controls['postprocessing']['spatial'].get('iterations', None) is not None:
                    self.raw_cfg.postProcessing.spatialFilter.numIterations = controls["postprocessing"]["spatial"]["iterations"]

            if controls['postprocessing'].get('temporal', None) is not None:
                if controls['postprocessing']['temporal'].get('enable', None) is not None:
                    self.raw_cfg.postProcessing.temporalFilter.enable = controls["postprocessing"]["temporal"]["enable"]
                if controls['postprocessing']['temporal'].get('persistency_mode', None) is not None:
                    if isinstance(controls["postprocessing"]["temporal"]["persistency_mode"], str):
                        controls["postprocessing"]["temporal"]["persistency_mode"] = getattr(dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode, controls["postprocessing"]["temporal"]["persistency_mode"])
                    self.raw_cfg.postProcessing.temporalFilter.persistencyMode = controls["postprocessing"]["temporal"]["persistency_mode"]
                if controls['postprocessing']['temporal'].get('alpha', None) is not None:
                    self.raw_cfg.postProcessing.temporalFilter.alpha = controls["postprocessing"]["temporal"]["alpha"]
                if controls['postprocessing']['temporal'].get('delta', None) is not None:
                    self.raw_cfg.postProcessing.temporalFilter.delta = controls["postprocessing"]["temporal"]["delta"]

            if controls['postprocessing'].get('threshold', None) is not None:
                if controls['postprocessing']['threshold'].get('min_range', None) is not None:
                    min_range = clamp(controls["postprocessing"]["threshold"]["min_range"], *LIMITS['range'])
                    self.raw_cfg.postProcessing.thresholdFilter.minRange = min_range
                if controls['postprocessing']['threshold'].get('max_range', None) is not None:
                    max_range = clamp(controls["postprocessing"]["threshold"]["max_range"], *LIMITS['range'])
                    self.raw_cfg.postProcessing.thresholdFilter.maxRange = max_range

            if controls['postprocessing'].get('brightness', None) is not None:
                if controls['postprocessing']['brightness'].get('min', None) is not None:
                    self.raw_cfg.postProcessing.brightnessFilter.minBrightness = controls["postprocessing"]["brightness"]["min"]
                if controls['postprocessing']['brightness'].get('max', None) is not None:
                    self.raw_cfg.postProcessing.brightnessFilter.maxBrightness = controls["postprocessing"]["brightness"]["max"]

            if controls['postprocessing'].get('speckle', None) is not None:
                if controls['postprocessing']['speckle'].get('enable', None) is not None:
                    self.raw_cfg.postProcessing.speckleFilter.enable = controls["postprocessing"]["speckle"]["enable"]
                if controls['postprocessing']['speckle'].get('range', None) is not None:
                    self.raw_cfg.postProcessing.speckleFilter.speckleRange = controls["postprocessing"]["speckle"]["range"]

            if controls['postprocessing'].get('decimation', None) is not None:
                if controls['postprocessing']['decimation'].get('factor', None) is not None:
                    self.raw_cfg.postProcessing.decimationFilter.decimationFactor = controls["postprocessing"]["decimation"]["factor"]
                if controls['postprocessing']['decimation'].get('mode', None) is not None:
                    if isinstance(controls["postprocessing"]["decimation"]["mode"], str):
                        controls["postprocessing"]["decimation"]["mode"] = getattr(dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode, controls["postprocessing"]["decimation"]["mode"])
                    self.raw_cfg.postProcessing.decimationFilter.decimationMode = controls["postprocessing"]["decimation"]["mode"]

        if controls.get('census_transform', None) is not None:
            if controls['census_transform'].get('kernel_size', None) is not None:
                if isinstance(controls["census_transform"]["kernel_size"], str):
                    controls["census_transform"]["kernel_size"] = getattr(dai.StereoDepthConfig.CensusTransform.KernelSize, controls["census_transform"]["kernel_size"])
                self.raw_cfg.censusTransform.kernelSize = controls["census_transform"]["kernel_size"]
            if controls['census_transform'].get('kernel_mask', None) is not None:
                self.raw_cfg.censusTransform.kernelMask = controls["census_transform"]["kernel_mask"]
            if controls['census_transform'].get('enable_mean_mode', None) is not None:
                self.raw_cfg.censusTransform.enableMeanMode = controls["census_transform"]["enable_mean_mode"]
            if controls['census_transform'].get('threshold', None) is not None:
                self.raw_cfg.censusTransform.threshold = controls["census_transform"]["threshold"]

        if controls.get('cost_matching', None) is not None:
            if controls['cost_matching'].get('disparity_width', None) is not None:
                if isinstance(controls["cost_matching"]["disparity_width"], str):
                    controls["cost_matching"]["disparity_width"] = getattr(dai.StereoDepthConfig.CostMatching.DisparityWidth, controls["cost_matching"]["disparity_width"])
                self.raw_cfg.costMatching.disparityWidth = controls["cost_matching"]["disparity_width"]
            if controls['cost_matching'].get('enable_companding', None) is not None:
                self.raw_cfg.costMatching.enableCompanding = controls["cost_matching"]["enable_companding"]
            if controls['cost_matching'].get('confidence_threshold', None) is not None:
                conf_threshold = clamp(controls["cost_matching"]["confidence_threshold"], *LIMITS['confidence_threshold'])
                self.raw_cfg.costMatching.confidenceThreshold = conf_threshold
            if controls['cost_matching'].get('linear_equation_parameters', None) is not None:
                if controls['cost_matching']['linear_equation_parameters'].get('alpha', None) is not None:
                    self.raw_cfg.costMatching.linearEquationParameters.alpha = controls["cost_matching"]["linear_equation_parameters"]["alpha"]
                if controls['cost_matching']['linear_equation_parameters'].get('beta', None) is not None:
                    self.raw_cfg.costMatching.linearEquationParameters.beta = controls["cost_matching"]["linear_equation_parameters"]["beta"]
                if controls['cost_matching']['linear_equation_parameters'].get('threshold', None) is not None:
                    self.raw_cfg.costMatching.linearEquationParameters.threshold = controls["cost_matching"]["linear_equation_parameters"]["threshold"]

        if controls.get('cost_aggregation', None) is not None:
            if controls['cost_aggregation'].get('division_factor', None) is not None:
                self.raw_cfg.costAggregation.divisionFactor = controls["cost_aggregation"]["division_factor"]
            if controls['cost_aggregation'].get('horizontal_penalty_cost_p1', None) is not None:
                self.raw_cfg.costAggregation.horizontalPenaltyCostP1 = controls["cost_aggregation"]["horizontal_penalty_cost_p1"]
            if controls['cost_aggregation'].get('horizontal_penalty_cost_p2', None) is not None:
                self.raw_cfg.costAggregation.horizontalPenaltyCostP2 = controls["cost_aggregation"]["horizontal_penalty_cost_p2"]
            if controls['cost_aggregation'].get('vertical_penalty_cost_p1', None) is not None:
                self.raw_cfg.costAggregation.verticalPenaltyCostP1 = controls["cost_aggregation"]["vertical_penalty_cost_p1"]
            if controls['cost_aggregation'].get('vertical_penalty_cost_p2', None) is not None:
                self.raw_cfg.costAggregation.verticalPenaltyCostP2 = controls["cost_aggregation"]["vertical_penalty_cost_p2"]

        ctrl.set(self.raw_cfg)
        self.queue.send(ctrl)

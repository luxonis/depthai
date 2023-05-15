import depthai as dai
from itertools import cycle
import logging

logger = logging.getLogger(__name__)

LIMITS = {
    'exposure_compensation': (-9, 9),
    'exposure': (1, 33000),
    'gain': (100, 1600),
    'focus': (0, 255),
    'white_balance': (1000, 12000),
    'brightness': (-10, 10),
    'contrast': (-10, 10),
    'saturation': (-10, 10),
    'sharpness': (0, 4),
    'luma_denoise': (0, 4),
    'chroma_denoise': (0, 4)
}

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

class CameraControl:
    def __init__(self):
        self.queue = None

        self._cycle_awb_mode = cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()])
        self._cycle_ab_mode = cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()])
        # self._cycle_effect_mode = cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])
        self._cycle_af_mode = cycle([item for name, item in vars(dai.CameraControl.AutoFocusMode).items() if name.isupper()])

        self._current_vals = {
            'exposure_time': 20000,
            'sensitivity': 300,
            'focus': 120,
            'sharpness': 0,
            'luma_denoise': 0,
            'chroma_denoise': 4,
        }

    def set_input_queue(self, queue: dai.DataInputQueue):
        self.queue = queue

    def switch_white_balance_mode(self):
        """
        Switch between auto white balance modes.
        """
        mode = next(self._cycle_awb_mode)
        logger.info(f'Setting white balance mode to {mode}.')
        self.send_controls({'white-balance': {'mode': mode}})

    def switch_anti_banding_mode(self):
        """
        Switch between anti-banding modes.
        """
        mode = next(self._cycle_ab_mode)
        logger.info(f'Setting anti-banding mode to {mode}.')
        self.send_controls({'ab_mode': mode})

    def switch_focus_mode(self):
        """
        Switch between focus modes.
        """
        mode = next(self._cycle_af_mode)
        logger.info(f'Setting focus mode to {mode}.')
        self.send_controls({'focus': {'mode': mode}})

    def exposure_time_up(self, step=500):
        """
        Increase exposure time by step.
        Args:
            step: In microseconds
        """
        if LIMITS['exposure'][1] < self._current_vals['exposure_time'] + step:
            logger.error(f'Exposure time cannot be greater than {LIMITS["exposure"][1]}')
            return
        self._current_vals['exposure_time'] += step
        self.send_controls({'exposure': {'manual': [self._current_vals['exposure_time'], self._current_vals['sensitivity']]}})
    def exposure_time_down(self, step=500):
        """
        Decrease exposure time by step.
        Args:
            step: In microseconds
        """
        if LIMITS['exposure'][0] > self._current_vals['exposure_time'] - step:
            logger.error(f'Exposure time cannot be less than {LIMITS["exposure"][0]}')
            return
        self._current_vals['exposure_time'] -= step
        self.send_controls({'exposure': {'manual': [self._current_vals['exposure_time'], self._current_vals['sensitivity']]}})

    def sensitivity_up(self, step=50):
        """
        Increase sensitivity by step.
        Args:
            step: In ISO
        """
        if LIMITS['gain'][1] < self._current_vals['sensitivity'] + step:
            logger.error(f'Sensitivity cannot be greater than {LIMITS["gain"][1]}')
            return
        self._current_vals['sensitivity'] += step
        self.send_controls({'exposure': {'manual': [self._current_vals['exposure_time'], self._current_vals['sensitivity']]}})

    def sensitivity_down(self, step=50):
        """
        Decrease sensitivity by step.
        Args:
            step: In ISO
        """
        if LIMITS['gain'][0] > self._current_vals['sensitivity'] - step:
            logger.error(f'Sensitivity cannot be less than {LIMITS["gain"][0]}')
            return
        self._current_vals['sensitivity'] -= step
        self.send_controls({'exposure': {'manual': [self._current_vals['exposure_time'], self._current_vals['sensitivity']]}})

    def focus_up(self, step=3):
        """
        Increase focus by step.
        """
        if LIMITS['focus'][1] < self._current_vals['focus'] + step:
            logger.error(f'Focus cannot be greater than {LIMITS["focus"][1]}')
            return
        self._current_vals['focus'] += step
        self.send_controls({'focus': {'manual': self._current_vals['focus']}})

    def focus_down(self, step=3):
        """
        Decrease focus by step.
        """
        if LIMITS['focus'][0] > self._current_vals['focus'] - step:
            logger.error(f'Focus cannot be less than {LIMITS["focus"][0]}')
            return
        self._current_vals['focus'] -= step
        self.send_controls({'focus': {'manual': self._current_vals['focus']}})

    def sharpness_up(self, step=1):
        """
        Increase sharpness by step
        """
        if LIMITS['sharpness'][1] < self._current_vals['sharpness'] + step:
            logger.error(f'Sharpness cannot be greater than {LIMITS["sharpness"][1]}')
            return
        self._current_vals['sharpness'] += step
        self.send_controls({'isp': {'sharpness': self._current_vals['sharpness']}})

    def sharpness_down(self, step=1):
        """
        Decrease sharpness by step
        """
        if LIMITS['sharpness'][0] > self._current_vals['sharpness'] - step:
            logger.error(f'Sharpness cannot be less than {LIMITS["sharpness"][0]}')
            return
        self._current_vals['sharpness'] -= step
        self.send_controls({'isp': {'sharpness': self._current_vals['sharpness']}})

    def luma_denoise_up(self, step=1):
        """
        Increase luma denoise by step
        """
        if LIMITS['luma_denoise'][1] < self._current_vals['luma_denoise'] + step:
            logger.error(f'Luma denoise cannot be greater than {LIMITS["luma_denoise"][1]}')
            return
        self._current_vals['luma_denoise'] += step
        self.send_controls({'isp': {'luma_denoise': self._current_vals['luma_denoise']}})

    def luma_denoise_down(self, step=1):
        """
        Decrease luma denoise by step
        """
        if LIMITS['luma_denoise'][0] > self._current_vals['luma_denoise'] - step:
            logger.error(f'Luma denoise cannot be less than {LIMITS["luma_denoise"][0]}')
            return
        self._current_vals['luma_denoise'] -= step
        self.send_controls({'isp': {'luma_denoise': self._current_vals['luma_denoise']}})

    def chroma_denoise_up(self, step=1):
        """
        Increase chroma denoise by step
        """
        if LIMITS['chroma_denoise'][1] < self._current_vals['chroma_denoise'] + step:
            logger.error(f'Chroma denoise cannot be greater than {LIMITS["chroma_denoise"][1]}')
            return
        self._current_vals['chroma_denoise'] += step
        self.send_controls({'isp': {'chroma_denoise': self._current_vals['chroma_denoise']}})

    def chroma_denoise_down(self, step=1):
        """
        Decrease chroma denoise by step
        """
        if LIMITS['chroma_denoise'][0] > self._current_vals['chroma_denoise'] - step:
            logger.error(f'Chroma denoise cannot be less than {LIMITS["chroma_denoise"][0]}')
            return
        self._current_vals['chroma_denoise'] -= step
        self.send_controls({'isp': {'chroma_denoise': self._current_vals['chroma_denoise']}})

    def send_controls(self, controls: dict = None):
        """
        Send controls to the camera. Dict structure and available options:

        {
            'exposure':{
                'auto': False, # setAutoExposureEnable
                'lock': False, # setAutoExposureLock
                'region': [0, 0.5, 0.3, 0.75], # setAutoExposureRegion
                'compensation': 0, # setExposureCompensation
                'manual': [1000, 100] # setManualExposure(exposureTimeUs, sensitivityIso [100..1600])
            },
            'ab_mode': 'auto', # either 'off', '50hz', '60hz', or 'auto'
            'focus': {
                'range': [0,255], # setAutoFocusLensRange
                'region': [0, 0.5, 0.3, 0.75], # setAutoFocusRegion
                'trigger': False, # setAutoFocusTrigger
                'mode': 'auto', # setAutoFocusMode, either 'auto', 'macro', 'continuous_video', 'continuous_picture', or 'edof'
                'manual': 0, # setManualFocus, 0..255
            },
            'white-balance': {
                'mode': 'auto', # setAutoWhiteBalanceMode, either 'off', 'auto', 'incandescent', 'fluorescent', 'warm-fluorescent', 'daylight', 'cloudy-daylight', 'twilight', or 'shade'
                'lock': False,
                'manual': 0, # setManualWhiteBalance, range 1000..12000 [colorTemperatureK]
            },
            'isp': {
                'brightness': 0, # setBrightness(), -10..10
                'contrast': 0, # setContrast(), -10..10
                'saturation': 0, # setSaturation(), -10..10
                'sharpness': 0, # setSharpness(), 0..4
                'luma_denoise': 0, # setLumaDenoise(), 0..4
                'chroma_denoise': 0, # setChromaDenoise(), 0..4
            },
            'still': True, # Capture still photo
            'reset': True, # Reset all
        }
        """
        if self.queue is None:
            logger.error('Cannot send controls when replaying.')
            return

        ctrl = dai.CameraControl()

        if controls.get('reset', None) and controls['reset']:
            logger.info('Resetting camera controls.')
            ctrl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)
            ctrl.setAutoExposureEnable(True)
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self.queue.send(ctrl)
            return

        if controls.get('exposure', None) is not None:
            if controls['exposure'].get('auto', False) and controls['exposure']['auto']:
                logger.info(f'Setting auto exposure to {controls["exposure"]["auto"]}.')
                ctrl.setAutoExposureEnable()
            if controls['exposure'].get('lock', None) is not None:
                logger.info(f'Setting exposure lock to {controls["exposure"]["lock"]}.')
                ctrl.setAutoExposureLock(controls['exposure']['lock'])
            if controls['exposure'].get('region', None) is not None:
                # logger.info(f'Setting exposure region to {controls["exposure"]["region"]}.')
                # ctrl.setAutoExposureRegion(*controls['exposure']['region'])
                logger.error('exposure region not yet implemented')
            if controls['exposure'].get('compensation', None) is not None:
                exp_comp = clamp(controls['exposure']['compensation'], *LIMITS['exposure_compensation'])
                logger.info(f'Setting exposure compensation to {exp_comp}.')
                ctrl.setAutoExposureCompensation(exp_comp)
            if controls['exposure'].get('manual', None) is not None:
                exp_time = clamp(controls['exposure']['manual'][0], *LIMITS['exposure'])
                exp_gain = clamp(controls['exposure']['manual'][1], *LIMITS['gain'])
                logger.info(f'Setting exposure to {exp_time}, gain to {exp_gain}.')
                ctrl.setManualExposure(exp_time, exp_gain)

        if controls.get('ab_mode', None) is not None:
            if isinstance(controls["ab_mode"], str):
                controls["ab_mode"] = getattr(dai.CameraControl.AntiBandingMode, controls["ab_mode"])
            logger.info(f'Setting anti-banding mode to {controls["ab_mode"]}.')
            ctrl.setAntiBandingMode(controls["ab_mode"])

        if controls.get('focus', None) is not None:
            if controls['focus'].get('range', None) is not None:
                logger.info(f'Setting focus range to {controls["focus"]["range"]}.')
                ctrl.setAutoFocusLensRange(*controls['focus']['range'])
            if controls['focus'].get('region', None) is not None:
                # logger.info(f'Setting focus region to {controls["focus"]["region"]}.')
                # ctrl.setAutoFocusRegion(*controls['focus']['region'])
                logger.error('focus region not yet implemented')
            if controls['focus'].get('trigger', None) is not None:
                if controls['focus']['trigger']:
                    logger.info('Auto-focus triggered.')
                    ctrl.setAutoFocusTrigger()
            if controls['focus'].get('mode', None) is not None:
                if isinstance(controls["focus"]["mode"], str):
                    controls["focus"]["mode"] = getattr(dai.CameraControl.AutoFocusMode, controls["focus"]["mode"])
                logger.info(f'Setting focus mode to {controls["focus"]["mode"]}.')
                ctrl.setAutoFocusMode(controls["focus"]["mode"])
            if controls['focus'].get('manual', None) is not None:
                focus = clamp(controls['focus']['manual'], *LIMITS['focus'])
                logger.info(f'Setting focus to {focus}.')
                ctrl.setManualFocus(focus)

        if controls.get('white-balance', None) is not None:
            if controls['white-balance'].get('mode', None) is not None:
                if isinstance(controls["focus"]["mode"], str):
                    controls["white-balance"]["mode"] = getattr(dai.CameraControl.AutoFocusMode, controls["white-balance"]["mode"])
                logger.info(f'Setting white balance mode to {controls["white-balance"]["mode"]}.')
                ctrl.setAutoWhiteBalanceMode(controls["white-balance"]["mode"])
            if controls['white-balance'].get('lock', None) is not None:
                logger.info(f'Setting white balance lock to {controls["white-balance"]["lock"]}.')
                ctrl.setAutoWhiteBalanceLock(controls['white-balance']['lock'])
            if controls['white-balance'].get('manual', None) is not None:
                wb_temp = clamp(controls['white-balance']['manual'], *LIMITS['white_balance'])
                logger.info(f'Setting white balance to {wb_temp}.')
                ctrl.setManualWhiteBalance(wb_temp)

        if controls.get('isp', None) is not None:
            if controls['isp'].get('brightness', None) is not None:
                brightness = clamp(controls['isp']['brightness'], *LIMITS['brightness'])
                logger.info(f'Setting brightness to {brightness}.')
                ctrl.setBrightness(brightness)
            if controls['isp'].get('contrast', None) is not None:
                contrast = clamp(controls['isp']['contrast'], *LIMITS['contrast'])
                logger.info(f'Setting contrast to {contrast}.')
                ctrl.setContrast(contrast)
            if controls['isp'].get('saturation', None) is not None:
                saturation = clamp(controls['isp']['saturation'], *LIMITS['saturation'])
                logger.info(f'Setting saturation to {saturation}.')
                ctrl.setSaturation(saturation)
            if controls['isp'].get('sharpness', None) is not None:
                sharpness = clamp(controls['isp']['sharpness'], *LIMITS['sharpness'])
                logger.info(f'Setting sharpness to {sharpness}.')
                ctrl.setSharpness(sharpness)
            if controls['isp'].get('luma_denoise', None) is not None:
                luma_denoise = clamp(controls['isp']['luma_denoise'], *LIMITS['luma_denoise'])
                logger.info(f'Setting luma denoise to {luma_denoise}.')
                ctrl.setLumaDenoise(luma_denoise)
            if controls['isp'].get('chroma_denoise', None) is not None:
                chroma_denoise = clamp(controls['isp']['chroma_denoise'], *LIMITS['chroma_denoise'])
                logger.info(f'Setting chroma denoise to {chroma_denoise}.')
                ctrl.setChromaDenoise(chroma_denoise)

        if controls.get('still', None) is not None:
            logger.info('Capturing still photo.')
            ctrl.setCaptureStill(controls['still'])

        self.queue.send(ctrl)

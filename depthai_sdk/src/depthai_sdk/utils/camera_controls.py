from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import cycle

import depthai as dai

from depthai_sdk.utils import get_logger

logger = get_logger(__name__)

AWB_MODE = cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()])
ANTI_BANDING_MODE = cycle(
    [item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()]
)
EFFECT_MODE = cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])

KEY2LOCKS = {
    ord('1'): 'awb_lock',
    ord('2'): 'ae_lock',
}

KEYS2ADJUSTABLE = {
    ord('3'): 'awb_mode',
    ord('4'): 'ae_compensation',
    ord('5'): 'antibanding_mode',
    ord('6'): 'effect_mode',
    ord('7'): 'brightness',
    ord('8'): 'contrast',
    ord('9'): 'saturation',
    ord('0'): 'sharpness',
    ord('['): 'luma_denoise',
    ord(']'): 'chroma_denoise'
}

KEYS2ACTION = {
    ord('h'): 'increase_dot_intensity',
    ord('j'): 'decrease_dot_intensity',
    ord('n'): 'increase_flood_intensity',
    ord('m'): 'decrease_flood_intensity',
    ord('='): 'increase',
    ord('+'): 'increase',
    ord('-'): 'decrease',
    ord('_'): 'decrease',
    ord('E'): 'auto_exposure',
    ord('F'): 'auto_focus_continuous',
    ord('T'): 'auto_focus_trigger',
    ord('b'): 'auto_white_balance',
    ord('.'): 'increase_focus',
    ord(','): 'decrease_focus',
    ord('o'): 'increase_exposure',
    ord('i'): 'decrease_exposure',
    ord('m'): 'increase_white_balance',
    ord('n'): 'decrease_white_balance',
    ord('l'): 'increase_iso',
    ord('k'): 'decrease_iso',
}

LIMITS = {
    'focus': (0, 255),
    'exposure': (1, 33000),
    'gain': (100, 1600),
    'white_balance': (1000, 12000),
    'ae_compensation': (-9, 9),
    'brightness': (-10, 10),
    'contrast': (-10, 10),
    'saturation': (-10, 10),
    'sharpness': (0, 4),
    'luma_denoise': (0, 4),
    'chroma_denoise': (0, 4)
}

STEPS = {
    'focus': 3,
    'exposure': 500,
    'gain': 50,
    'white_balance': 200
}


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


@dataclass
class ControlConfig:
    awb_lock: bool = field(default=False)
    ae_lock: bool = field(default=False)

    focus: int = field(default=150)
    exposure: int = field(default=20000)
    gain: int = field(default=800)
    white_balance: int = field(default=4000)
    ae_compensation: int = field(default=0)
    brightness: int = field(default=0)
    contrast: int = field(default=0)
    saturation: int = field(default=0)
    sharpness: int = field(default=0)
    luma_denoise: int = field(default=0)
    chroma_denoise: int = field(default=0)


class CameraControls:
    def __init__(self, control_queue: dai.DataInputQueue):
        self.selected_control = None
        self.control_queue = control_queue
        self.control_config = asdict(ControlConfig())

        # Iterators
        self.modes = {
            'awb_mode': AWB_MODE,
            'antibanding_mode': ANTI_BANDING_MODE,
            'effect_mode': EFFECT_MODE
        }

        # Selected modes
        self.selected_modes = {
            'awb_mode': next(self.modes['awb_mode']),
            'antibanding_mode': next(self.modes['antibanding_mode']),
            'effect_mode': next(self.modes['effect_mode']),
        }

    def send_controls(self, controls: dict = None):
        """
        Send controls to the camera.
        """
        controls = controls or self.control_config

        if controls.get('exposure', None) or controls.get('gain', None):
            logger.info(
                f'Setting exposure to {self.control_config["exposure"]}, gain to {self.control_config["gain"]}.')
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(self.control_config['exposure'], self.control_config['gain'])
            self.control_queue.send(ctrl)

        if controls.get('focus', None):
            logger.info(f'Setting focus to {self.control_config["focus"]}.')
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(self.control_config['focus'])
            self.control_queue.send(ctrl)

        if controls.get('white_balance', None):
            logger.info(f'Setting white balance to {self.control_config["white_balance"]}.')
            ctrl = dai.CameraControl()
            ctrl.setManualWhiteBalance(self.control_config['white_balance'])
            self.control_queue.send(ctrl)

        if controls.get('ae_compensation', None):
            logger.info(f'Setting AE compensation to {self.control_config["ae_compensation"]}.')
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureCompensation(self.control_config['ae_compensation'])
            self.control_queue.send(ctrl)

        if controls.get('brightness', None):
            logger.info(f'Setting brightness to {self.control_config["brightness"]}.')
            ctrl = dai.CameraControl()
            ctrl.setBrightness(self.control_config['brightness'])
            self.control_queue.send(ctrl)

        if controls.get('contrast', None):
            logger.info(f'Setting contrast to {self.control_config["contrast"]}.')
            ctrl = dai.CameraControl()
            ctrl.setContrast(self.control_config['contrast'])
            self.control_queue.send(ctrl)

        if controls.get('saturation', None):
            logger.info(f'Setting saturation to {self.control_config["saturation"]}.')
            ctrl = dai.CameraControl()
            ctrl.setSaturation(self.control_config['saturation'])
            self.control_queue.send(ctrl)

        if controls.get('luma_denoise', None):
            logger.info(f'Setting luma denoise to {self.control_config["luma_denoise"]}.')
            ctrl = dai.CameraControl()
            ctrl.setLumaDenoise(self.control_config['luma_denoise'])
            self.control_queue.send(ctrl)

        if controls.get('chroma_denoise', None):
            logger.info(f'Setting chroma denoise to {self.control_config["chroma_denoise"]}.')
            ctrl = dai.CameraControl()
            ctrl.setChromaDenoise(self.control_config['chroma_denoise'])
            self.control_queue.send(ctrl)

        if controls.get('sharpness', None):
            logger.info(f'Setting sharpness to {self.control_config["sharpness"]}.')
            ctrl = dai.CameraControl()
            ctrl.setSharpness(self.control_config['sharpness'])
            self.control_queue.send(ctrl)

        if controls.get('awb_mode', None):
            logger.info(f'Setting AWB mode to {self.selected_modes["awb_mode"]}.')
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(self.selected_modes['awb_mode'])
            self.control_queue.send(ctrl)

        if controls.get('antibanding_mode', None):
            logger.info(f'Setting antibanding mode to {self.selected_modes["antibanding_mode"]}.')
            ctrl = dai.CameraControl()
            ctrl.setAntiBandingMode(self.selected_modes['antibanding_mode'])
            self.control_queue.send(ctrl)

        if controls.get('effect_mode', None):
            logger.info(f'Setting effect mode to {self.selected_modes["effect_mode"]}.')
            ctrl = dai.CameraControl()
            ctrl.setEffectMode(self.selected_modes['effect_mode'])
            self.control_queue.send(ctrl)

    def send_lock_controls(self, controls: dict):
        if controls.get('awb_lock', None):
            logger.info(f'AWB lock: {self.control_config["awb_lock"]}')
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceLock(self.control_config['awb_lock'])
            self.control_queue.send(ctrl)

        if controls.get('ae_lock', None):
            logger.info(f'AE lock: {self.control_config["ae_lock"]}')
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(self.control_config['ae_lock'])
            self.control_queue.send(ctrl)

    def send_auto_controls(self, controls: dict):
        """
        Send controls related to automatic settings to the camera.
        """
        controls = defaultdict(lambda: None, controls)

        if controls.get('auto_white_balance', None):
            logger.info('Setting AWB to auto.')
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self.control_queue.send(ctrl)

        if controls.get('auto_exposure', None):
            logger.info('Setting AE to auto.')
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self.control_queue.send(ctrl)

        if controls.get('auto_focus_trigger', None):
            logger.info('Setting AF to auto.')
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            self.control_queue.send(ctrl)

        if controls.get('auto_focus_continuous', None):
            logger.info('Setting AF to continuous.')
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            self.control_queue.send(ctrl)

        if controls.get('auto_anti_banding', None):
            logger.info('Setting anti-banding to auto.')
            ctrl = dai.CameraControl()
            ctrl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)
            self.control_queue.send(ctrl)

    def send_controls_by_key(self, key: int) -> None:
        if key == -1:
            return

        self.selected_control = KEYS2ADJUSTABLE.get(key, None) or self.selected_control

        action = KEYS2ACTION.get(key, None)
        lock = KEY2LOCKS.get(key, None)

        if lock is not None:
            self.adjust_lock(lock)
            self.send_lock_controls({lock: True})

        if key in KEYS2ADJUSTABLE and not action:
            logger.info(f'Selected control: {self.selected_control}')

        if not action:
            return

        if action == 'increase':
            self.adjust_control(step=1)
            self.send_controls({self.selected_control: True})
        elif action == 'decrease':
            self.adjust_control(step=-1)
            self.send_controls({self.selected_control: True})
        elif action.startswith('auto'):
            self.adjust_control(step=None)
            self.send_auto_controls({action: True})
        elif action == 'increase_focus':
            self.adjust_lens_position(step=1)
        elif action == 'decrease_focus':
            self.adjust_lens_position(step=-1)
        elif action == 'increase_exposure':
            self.__adjust_setting('exposure')
            self.send_controls({'gain': True})
        elif action == 'decrease_exposure':
            self.__adjust_setting('exposure', positive=False)
            self.send_controls({'exposure': True})
        elif action == 'increase_iso':
            self.__adjust_setting('gain')
            self.send_controls({'gain': True})
        elif action == 'decrease_iso':
            self.__adjust_setting('gain', positive=False)
            self.send_controls({'gain': True})
        elif action == 'increase_white_balance':
            self.__adjust_setting('white_balance')
            self.send_controls({'white_balance': True})
        elif action == 'decrease_white_balance':
            self.__adjust_setting('white_balance', positive=False)
            self.send_controls({'white_balance': True})

    def adjust_lock(self, lock: str) -> None:
        if lock in self.control_config:
            self.control_config[lock] = not self.control_config[lock]
        else:
            self.control_config[lock] = True

    def adjust_control(self, step: int = None) -> None:
        if self.selected_control in self.selected_modes:
            self.selected_modes[self.selected_control] = next(self.modes[self.selected_control])
        elif step:
            self.__adjust_setting(self.selected_control, step, positive=step > 0)

    def adjust_lens_position(self, step: int) -> None:
        self.__adjust_setting('focus', step, positive=step > 0)
        self.send_controls({'focus': True})

    def __adjust_setting(self, key: str, step: int = 0, positive: bool = True) -> None:
        if key in STEPS:
            step = STEPS[key] * (1 if positive else -1)

        if not self.control_config.get(key, None):
            self.control_config[key] = step
        else:
            self.control_config[key] += step

        if key in LIMITS:
            self.control_config[key] = clamp(self.control_config[key], *LIMITS[key])

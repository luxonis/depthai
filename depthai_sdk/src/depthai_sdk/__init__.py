from depthai_sdk.args_parser import ArgsParser
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.constants import CV2_HAS_GUI_SUPPORT
from depthai_sdk.logger import set_logging_level
from depthai_sdk.oak_camera import OakCamera
from depthai_sdk.oak_device import OakDevice
from depthai_sdk.previews import *
from depthai_sdk.record import *
from depthai_sdk.replay import *
from depthai_sdk.utils import *
from depthai_sdk.utils import _create_config, get_config_field
from depthai_sdk.visualize import *

__version__ = '1.12.1'


def __import_sentry(sentry_dsn: str) -> None:
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=1.0,
            release=f'depthai_sdk@{__version__}',
            with_locals=False,
        )
    except:
        pass


config_exists = False
# Check if sentry is enabled
try:
    sentry_status = get_config_field('sentry')
    config_exists = True
except FileNotFoundError:
    sentry_status = False

if config_exists and sentry_status:
    sentry_dsn = get_config_field('sentry_dsn')
    __import_sentry(sentry_dsn)
elif not config_exists:
    _create_config()

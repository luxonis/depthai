from depthai_sdk.args_parser import ArgsParser
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.constants import CV2_HAS_GUI_SUPPORT
from depthai_sdk.logger import set_logging_level
from depthai_sdk.oak_camera import OakCamera
from depthai_sdk.previews import *
from depthai_sdk.record import *
from depthai_sdk.replay import *
from depthai_sdk.utils import *
from depthai_sdk.utils import _create_config, get_config_field, _sentry_before_send
from depthai_sdk.visualize import *

__version__ = '1.15.1'


def __import_sentry(sentry_dsn: str) -> None:
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=1.0,
            release=f'depthai_sdk@{__version__}',
            with_locals=False,
            before_send=_sentry_before_send
        )
    except:
        pass


sentry_dsn = get_config_field('sentry_dsn')
sentry_status = get_config_field('sentry')
if sentry_dsn and sentry_status:
    __import_sentry(sentry_dsn)

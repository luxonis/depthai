import logging

__all__ = ['set_logging_level']


def _configure_logger():
    """
    Configure the logging module.
    """
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def set_logging_level(level):
    """
    Set the logging level for the root logger.
    """
    logging.getLogger().setLevel(level)


_configure_logger()

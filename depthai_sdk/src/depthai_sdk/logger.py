import logging

__all__ = ['set_logging_level']

LOGGER = logging.getLogger(__name__)
"""The DepthAI SDK logger."""

def _configure_logger():
    """
    Configure the logging module.
    """
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

def set_logging_level(level):
    """
    Set the logging level for the DepthAI SDK logger.
    """
    LOGGER.setLevel(level)


_configure_logger()

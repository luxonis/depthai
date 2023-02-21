import logging

LEVEL = logging.INFO


def set_logging_level(level):
    global LEVEL
    LEVEL = level


def get_logger(name, level=None):
    logger = logging.getLogger(name)
    logger.setLevel(level or LEVEL)
    handler = logging.StreamHandler()
    handler.setLevel(level or LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

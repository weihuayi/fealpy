import logging

__version__ = '3.0.4'

logger = logging.getLogger('fealpy')
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
    logger.propagate = False

"""FEALPy: Finite Element Analysis Library in Python
====
"""
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
    logger.propagate = False

__version__ = "2.0.0"

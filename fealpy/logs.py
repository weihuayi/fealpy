import sys
from tqdm import tqdm
import logging

FORMAT = '%(asctime)s %(levelname)s - %(message)s'
FORMATTER = logging.Formatter(FORMAT, datefmt="%d-%m-%Y %H:%M:%S")

logger = logging.getLogger('fealpy')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(FORMATTER)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)
logger.propagate = False


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.setFormatter(FORMATTER)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

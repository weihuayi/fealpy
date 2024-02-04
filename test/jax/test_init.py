
import pytest
from fealpy.jax import logger

def test_logger():
    logger.warning("This is a test warning in fealpy.jax!")

if __name__ == "__main__":
    test_logger()

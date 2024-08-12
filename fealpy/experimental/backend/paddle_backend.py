from typing import Optional, Union, Tuple
from functools import reduce
from math import factorial
from itertools import combinations_with_replacement

import numpy as np
import paddle
from cupy.typing import NDArray
from cupy.linalg import det

from .base import (
    Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING
)


class CuPyBackend(Backend[NDArray], backend_name='cupy'):
    DATA_CLASS = cp.ndarray

    linalg = cp.linalg
    random = cp.random

    @staticmethod
    def context(x):
        return {"dtype": x.dtype, "device": x.device}

    @staticmethod
    def set_default_device(device) -> None:
        raise NotImplementedError("`set_default_device` is not supported by NumPyBackend")

    @staticmethod
    def get_device(x, /):
        return x.device 

    @staticmethod
    def to_numpy(x: NDArray, /) -> np.ndarray:
        return cp.ndarray.get(x) 

    @staticmethod
    def from_numpy(x: np.ndarray, /) -> NDArray:
        return cp.array(x) 

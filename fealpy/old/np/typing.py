from typing import Tuple, Union, Literal, Callable
from functools import reduce

import numpy as np

### Types

Array = np.ndarray
_dtype = np.dtype

# NumPy doesn't have specific device types, so we'll use a generic string
DeviceLike = str

_bool = np.bool_
_int = np.int_
_int8 = np.int8
_int16 = np.int16
_int32 = np.int32
_int64 = np.int64
_float = np.float_
_float16 = np.float16
_float32 = np.float32
_float64 = np.float64
_uint8 = np.uint8

Number = Union[int, float]
Index = Union[int, slice, Tuple[int, ...], Array]
EntityName = Literal['cell', 'face', 'edge', 'node']
_int_func = Callable[..., int]
Barycenters = Union[Tuple[Array, ...], Array]

### Constants

_S = slice(None)

class Size(Tuple[int, ...]):
    def numel(self) -> int:
        return reduce(lambda x, y: x * y, self, 1)


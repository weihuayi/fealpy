import taichi as ti
from typing import Union, Tuple

from ..utils import numpy_to_taichi_dtype


@ti.data_oriented
class COOMatrix():

    def __init__(self, arg1, shape: Tuple[int, int], dtype=None, itype=None,
                 copy=False):
        pass

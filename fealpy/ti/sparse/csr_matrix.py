import numpy as np
import taichi as ti


@ti.data_oriented
class CSRMatrix():
    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if len(arg1) == 3:
            self.data, self.indices, self.indptr = arg1
        else:
            rasie ValueError(f"Now, we just support arg1 == (data, indices, indptr)!")

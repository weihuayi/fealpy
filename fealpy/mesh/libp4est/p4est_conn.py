# p4est_conn.py
from mpi4py.MPI import Comm, _addressof
import numpy as np
import ctypes
from ._p4est import (
    get_num_vertices_py,
    get_vertices_ptr_py,
    get_vertices_shape_py,
    p4est_conn_new_unitsquare_py,

    p4est_new_py,
)

class P4estConnectivity:
    def __init__(self, capsule=None):
        self._capsule = capsule

    @classmethod
    def new_unitsquare(cls):
        return cls(p4est_conn_new_unitsquare_py())

    @property
    def num_vertices(self):
        return get_num_vertices_py(self._capsule)

    @property
    def vertices(self):
        ptr = get_vertices_ptr_py(self._capsule)
        if ptr == 0:
            return None
        shape = get_vertices_shape_py(self._capsule)

        # 使用 ctypes 构造数组
        buffer_type = ctypes.c_double * (shape[0] * shape[1])
        buffer = buffer_type.from_address(ptr)

        # 转换为 NumPy 数组并设置只读
        arr = np.ctypeslib.as_array(buffer).reshape(shape)
        arr.flags.writeable = False
        return arr

    def __del__(self):
        if hasattr(self, '_capsule'):
            del self._capsule


class P4est():
    def __init__(self, mpicomm: Comm, connectivity: P4estConnectivity):
        conn_cap = connectivity._capsule
        self._capsule = p4est_new_py(_addressof(mpicomm), conn_cap)

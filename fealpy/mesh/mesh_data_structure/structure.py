
from numpy.typing import NDArray
from numpy import dtype
import numpy as np

from .mesh_ds import MeshDataStructure


class Structure(MeshDataStructure):
    """
    @brief The base class for all structure meshes.

    Structure or Non-structure only changes the initializing method of mesh, but\
    does not implement any abstract method, or define any class constants.
    """
    def __init__(self, *nx: int, itype: dtype) -> None:
        if len(nx) != self.TD:
            raise ValueError(f"Number of `nx` must match the top dimension.")
        self.nx_ = np.array(nx, dtype=itype)
        self.NN = np.prod(self.nx_ + 1)
        self.itype = itype

    @property
    def nx(self):
        return self.nx_[0]
    @property
    def ny(self):
        return self.nx_[1]
    @property
    def nz(self):
        return self.nx_[2]

    @property
    def cell(self):
        TD = self.TD
        NN = self.NN
        NC = np.prod(self.nx_)
        cell = np.zeros((NC, 2*NC), dtype=self.itype)
        idx = np.arange(NN).reshape(self.nx_+1)
        c = idx[(slice(-1), )*TD]
        cell[:, 0] = c.flat

        ## This is for any topology dimension:

        # for i in range(1, TD + 1):
        #     begin = 2**(i-1)
        #     end = 2**i
        #     jump = np.prod(self._nx+1)//(self.nx+1)
        #     cell[:, begin:end] = cell[:, 0:end-begin] + jump

        if TD >= 1:
            cell[:, 1:2] = cell[:, 0:1] + 1

        if TD >= 2:
            cell[:, 2:4] = cell[:, 0:2] + self.ny + 1

        if TD >= 3:
            cell[:, 4:8] = cell[:, 0:4] + (self.ny+1)*(self.nz+1)

        return cell


class Nonstructure(MeshDataStructure):
    """
    @brief
    """
    def __init__(self, NN: int, cell: NDArray) -> None:
        self.reinit(NN=NN, cell=cell)

    def reinit(self, NN: int, cell: NDArray):
        self.NN = NN
        self.cell = cell
        self.itype = cell.dtype
        self.construct()

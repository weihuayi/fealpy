
__all__ = ['LinearMeshCFEDof']

from typing import Union, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..mesh.mesh_base import Mesh


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, NDArray]
_S = slice(None)


class LinearMeshCFEDof(Generic[_MT]):
    def __init__(self, mesh: _MT, p: int):
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, TD)

    def is_boundary_dof(self, threshold=None):
        TD = self.mesh.top_dimension()
        gdof = self.number_of_global_dofs()
        if isinstance(threshold, np.ndarray):
            index = threshold
            if (index.dtype == np.bool_) and (len(index) == gdof):
                return index
        else:
            index = self.mesh.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof] = True
        return isBdDof

    def edge_to_dof(self, index: Index=_S):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    def face_to_dof(self, index: Index=_S):
        return self.mesh.face_to_ipoint(self.p, index=index)

    def cell_to_dof(self, index: Index=_S):
        return self.mesh.cell_to_ipoint(self.p, index=index)

    def interpolation_points(self, index: Index=_S) -> NDArray:
        return self.mesh.interpolation_points(self.p, index=index)

    def number_of_global_dofs(self) -> int:
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)


from typing import Union, TypeVar

import torch
from torch import Tensor

from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof


_dtype = torch.dtype
_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, Tensor]
_S = slice(None)


class LagrangeFESpace(FunctionSpace[_MT]):
    def __init__(self, mesh: _MT, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.ds.itype
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def number_of_local_dofs(self, doftype='cell'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self):
        return self.dof.cell2dof

    def face_to_dof(self):
        return self.dof.face_to_dof()

    def basis(self, bc, index: Index=_S, variable='u'):
        return self.mesh.shape_function(bc, p=self.p, variable=variable)

    def grad_basis(self, bc, index: Index=_S, variable='u'):
        """
        @brief
        """
        return self.mesh.grad_shape_function(bc, p=self.p, index=index, variable=variable)

    def hess_basis(self, bc, index: Index=_S, variable='u'):
        """
        @brief
        """
        return self.mesh.hess_shape_function(bc, p=self.p, index=index, variable=variable)


    def value(self, uh, bc, index: Index=_S):
        """
        @brief
        """
        pass


from typing import Union, TypeVar, Generic, Callable

import torch
from torch import Tensor

from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, Tensor]
_S = slice(None)


class LagrangeFESpace(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.ds.itype
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def number_of_local_dofs(self, doftype='cell'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self):
        return self.dof.cell_to_dof()

    def face_to_dof(self):
        return self.dof.face_to_dof()

    def is_boundary_dof(self, threshold=None):
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")

    def interpolate(self, gD: Union[Callable[..., Tensor], Tensor],
                    uh: Tensor,
                    index: Index=_S):
        ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
        GD = self.mesh.geo_dimension()

        if callable(gD):
            gD = gD(ipoints[index])

        if (len(uh.shape) == 1) or (self.doforder == 'vdims'):
            if len(uh.shape) == 1 and gD.shape[-1] == 1:
                gD = gD.squeeze(-1)
            uh[index] = gD

        elif self.doforder == 'sdofs':
            if isinstance(gD, (int, float)):
                uh[..., index] = gD
            elif isinstance(gD, Tensor):
                if gD.shape == (GD, ):
                    uh[..., index] = gD[:, None]
                else:
                    uh[..., index] = gD.T
            else:
                raise ValueError("Unsupported type for gD. Must be a callable, int, float, or Tensor.")

        return uh

    def basis(self, bc: Tensor, index: Index=_S, variable='u'):
        return self.mesh.shape_function(bc, self.p, index=index, variable=variable)

    def grad_basis(self, bc: Tensor, index: Index=_S, variable='u'):
        """
        @brief
        """
        return self.mesh.grad_shape_function(bc, self.p, index=index, variable=variable)

    def hess_basis(self, bc: Tensor, index: Index=_S, variable='u'):
        """
        @brief
        """
        return self.mesh.hess_shape_function(bc, self.p, index=index, variable=variable)

    def value(self, uh: Tensor, bc: Tensor, index: Index=_S):
        """
        @brief
        """
        pass

    def grad(self, uh: Tensor, bc: Tensor, index: Index=_S):
        pass

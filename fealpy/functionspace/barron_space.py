
from typing import TypeVar, Generic, Callable, NoReturn

from ..typing import TensorLike, Index, _S, Threshold
from ..backend import bm
from ..decorator import barycentric, cartesian
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function


_MT = TypeVar('_MT', bound=Mesh)


class BarronSpace(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int = 1):
        self.mesh = mesh
        self.p = p

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def __str__(self):
        return "Barron space on linear mesh"

    def number_of_local_dofs(self, doftype = 'cell') -> int:
        return self.p

    def number_of_global_dofs(self) -> int:
        return self.mesh.number_of_cells() * self.p

    def interpolation_points(self) -> TensorLike:
        raise TypeError("Barron space does not support interpolation points")

    def cell_to_dof(self, index: Index = _S) -> TensorLike:
        NC = self.mesh.number_of_cells()
        cell2dof = bm.arange(0, NC*self.p, self.p, dtype=self.itype, device=self.device)
        STEP = bm.arange(self.p, dtype=self.itype, device=self.device)
        return cell2dof[index, None] + STEP[None, :]

    def face_to_dof(self, index: Index = _S) -> TensorLike:
        return self.cell_to_dof(index=index)

    def edge_to_dof(self, index=_S):
        raise TypeError("Barron space does not support dofs on edges")

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        raise TypeError("Barron space does not support boundary dofs")

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def project(self, u: Callable[..., TensorLike] | TensorLike) -> NoReturn:
        raise TypeError("Barron space does not support projection.")

    def interpolate(self, u: Callable[..., TensorLike] | TensorLike) -> NoReturn:
        raise TypeError("Barron space does not support interpolation.")

    def boundary_interpolate(self,
        gd: Callable | int | float | TensorLike,
        uh: TensorLike | None = None,
        *,
        threshold: Threshold | None = None,
        method = None
    ) -> NoReturn:
        """Barron space does not support interpolation."""
        raise TypeError("Barron space does not support interpolation.")

    set_dirichlet_bc = boundary_interpolate

    def basis(self, bc: TensorLike, index: Index = _S):
        phi = self.mesh.shape_function(bc, 1, index=index) # (..., NQ, ldof)

        if self.p == 1:
            return phi
        else:
            raise NotImplementedError

    face_basis = basis

    def grad_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.grad_shape_function(bc, self.p, index=index, variables=variable)

    def hess_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.hess_shape_function(bc, self.p, index=index, variables=variable)

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike: 
        phi = self.basis(bc, index=index)
        e2dof = self.cell_to_dof(index=index)
        val = bm.einsum('cql, ...cl -> ...cq', phi, uh[..., e2dof])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        gphi = self.grad_basis(bc, index=index)
        e2dof = self.cell_to_dof(index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[e2dof])
        return val

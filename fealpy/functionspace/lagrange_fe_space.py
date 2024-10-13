
from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from fealpy.decorator import barycentric, cartesian


_MT = TypeVar('_MT', bound=Mesh)


class LagrangeFESpace(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)
        elif ctype == 'D':
            self.dof = LinearMeshDFEDof(mesh, p)
        else:
            raise ValueError(f"Unknown type: {ctype}")

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        # self.multi_index_matrix = mesh.multi_index_matrix(p,2)

        #TODO:JAX
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def __str__(self):
        return "Lagrange finite element space on linear mesh!"

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self) -> int:
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        return self.dof.interpolation_points()

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.cell_to_dof()[index]

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof()[index]

    def edge_to_dof(self, index=_S):
        return self.dof.edge_to_dof()[index]

    def is_boundary_dof(self, threshold=None) -> TensorLike:
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        assert callable(u)

        if not hasattr(u, 'coordtype'):
            ips = self.interpolation_points()
            uI = u(ips)
        else:
            if u.coordtype == 'cartesian':
                ips = self.interpolation_points()
                uI = u(ips)
            elif u.coordtype == 'barycentric':
                TD = self.TD
                p = self.p
                bcs = self.mesh.multi_index_matrix(p, TD)/p
                uI = u(bcs)
        return uI

    def boundary_interpolate(self,
            gD: Union[Callable, int, float, TensorLike],
            uh: TensorLike,
            threshold: Optional[Threshold]=None) -> TensorLike:
        """Set the first type (Dirichlet) boundary conditions.

        Parameters:
            gD: boundary condition function or value (can be a callable, int, float, TensorLike).
            uh: TensorLike, FE function uh .
            threshold: optional, threshold for determining boundary degrees of freedom (default: None).

        Returns:
            TensorLike: a bool array indicating the boundary degrees of freedom.

        This function sets the Dirichlet boundary conditions for the FE function `uh`. It supports
        different types for the boundary condition `gD`, such as a function, a scalar, or a array.
        """
        ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
        isDDof = self.is_boundary_dof(threshold=threshold)

        if callable(gD):
            gD = gD(ipoints[isDDof])

        uh = bm.set_at(uh, (..., isDDof), gD)
        return uh, isDDof

    set_dirichlet_bc = boundary_interpolate

    def basis(self, bc: TensorLike, index: Index=_S):
        phi = self.mesh.shape_function(bc, self.p, index=index)
        return phi[None, ...] # (NC, NQ, LDOF)

    def grad_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.grad_shape_function(bc, self.p, index=index, variables=variable)

    def hess_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.hess_shape_function(bc, self.p, index=index, variables=variable)

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        phi = self.basis(bc, index=index)
        e2dof = self.dof.cell_to_dof(index=index)
        val = bm.einsum('cql, ...cl -> ...cq', phi, uh[..., e2dof])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell_to_dof(index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[cell2dof])
        return val[...]

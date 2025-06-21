from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof
from .function import Function
from fealpy.decorator import barycentric, cartesian


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)
_F = Union[Callable[..., TensorLike], TensorLike, Number]


class ParametricLagrangeFESpace(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p, q=None, ctype='C'):
        self.mesh = mesh
        self.p = p
        
        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh.linearmesh, p)
        elif ctype == 'D':
            self.dof = LinearMeshDFEDof(mesh.linearmesh, p)
        else:
            raise ValueError(f"Unknown type: {ctype}")

        self.cellmeasure = mesh.cell_area()
        self.multi_index_matrix = mesh.multi_index_matrix

        self.device = mesh.device
        self.GD = mesh.geo_dimension()
        self.TD = mesh.top_dimension()
        
        q = q if q is not None else p+3 
        self.quadrature_formula = self.mesh.quadrature_formula(q, etype='cell')
        
        self.itype = mesh.itype
        self.ftype = mesh.ftype
     
    def __str__(self):
         return "Parametric Lagrange finite element space!"

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof.number_of_local_dofs(doftype=doftype)
    
    def number_of_global_dofs(self) -> int:
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        return self.mesh.interpolation_points(self.p)

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.cell_to_dof()[index]

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof()[index]

    def edge_to_dof(self, index: Index=_S):
        return self.dof.edge_to_dof()[index]
    
    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold=threshold, method=method)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous space.")

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD
    
    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        assert callable(u)

        if not hasattr(u, 'coordtype'):
            ips = self.mesh.node
            uI = u(ips)
        else:
            if u.coordtype == 'cartesian':
                ips = self.mesh.node
                uI = u(ips)
            elif u.coordtype == 'barycentric':
                TD = self.TD
                p = self.p
                bcs = self.mesh.multi_index_matrix(p, TD)/p # (NQ, TD+1)
                val = u(bcs) # (NC, ldof)
                cell2dof = self.cell_to_dof()
                uI = bm.zeros(self.number_of_global_dofs(), dtype=sellf.ftype)
                uI = bm.set_at(uI, cell2dof, val)
        return self.function(uI)

    def boundary_interpolate(self,
            gd: Union[Callable, int, float, TensorLike],
            uh: Optional[TensorLike] = None,
            *, threshold: Optional[Threshold]=None, method=None) -> TensorLike:
        """Set the first type (Dirichlet) boundary conditions.

        Parameters:
            gd: boundary condition function or value (can be a callable, int, float, TensorLike).
            uh: TensorLike, FE function uh .
            threshold: optional, threshold for determining boundary degrees of freedom (default: None).

        Returns:
            TensorLike: a bool array indicating the boundary degrees of freedom.

        This function sets the Dirichlet boundary conditions for the FE function `uh`. It supports
        different types for the boundary condition `gd`, such as a function, a scalar, or a array.
        """
        ipoints = self.mesh.node
        isDDof = self.is_boundary_dof(threshold=threshold, method='interp')
        if bm.is_tensor(gd):
            assert len(gd) == self.number_of_global_dofs()
            if uh is None:
                uh = bm.zeros_like(gd)
            uh[isDDof] = gd[isDDof] 
            return uh,isDDof 
        if callable(gd):
            gd = gd(ipoints[isDDof])
        if uh is None:
            uh = self.function()
        uh[:] = bm.set_at(uh[:], (..., isDDof), gd)

        return self.function(uh), isDDof

    set_dirichlet_bc = boundary_interpolate

    def project(self, u: Union[Callable[..., TensorLike], TensorLike], M=None) -> TensorLike:
        """
        """
        pass
 
    @barycentric
    def edge_basis(self, bc: TensorLike):
        phi = self.mesh.shape_function(bc)
        return phi 

    @barycentric
    def face_basis(self, bc: TensorLike):
        phi = self.mesh.shape_function(bc)
        return phi 

    @barycentric
    def basis(self, bc: TensorLike, index: Index=_S):
        """
        @berif 计算基函数在重心坐标点处的函数值，注意 bc 的形状为 (..., TD+1), TD 为 bc
        所在空间的拓扑维数。
        """
        p = self.p
        phi = self.mesh.shape_function(bc, p=p, variables='x')
        return phi 

    @barycentric
    def grad_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        """
        @berif 计算空间基函数关于实际坐标点 x 的梯度。
        """
        p = self.p
        gphi = self.mesh.grad_shape_function(bc, index=index, p=p, variables=variable)
        return gphi

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S):
        phi = self.basis(bc) #
        cell2dof = self.dof.cell_to_dof()[index]
        dim = len(uh.shape) - 1
        val = bm.einsum('cql, cl -> cq', phi, uh[cell2dof])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index:Index=_S):
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell_to_dof()[index]
        dim = len(uh.shape) - 1
        val = bm.einsum('cqlm, cl -> cqm', gphi, uh[cell2dof])
        return val

    def integral_basis(self, q=None):
        cell2dof = self.cell_to_dof()
        qf = self.quadrature_formula if q is None else self.mesh.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        rm = self.mesh.reference_cell_measure()
        J = self.mesh.jacobi_matrix(bcs)
        G = self.mesh.first_fundamental_form(J)
        d = bm.sqrt(bm.linalg.det(G))  #(NC, NQ)
        phi = self.basis(bcs)  #(NC, NQ, ldof)
        cc = bm.einsum('q, cqi, cq -> ci', ws*rm, phi, d)
        gdof = self.number_of_global_dofs()
        c = bm.zeros(gdof, dtype=self.ftype)
        bm.add.at(c, cell2dof, cc)
        return c


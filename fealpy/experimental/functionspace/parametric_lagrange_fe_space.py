
from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S

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
    def __init__(self, mesh: _MT, p, q=None, spacetype='C'):
        self.mesh = mesh
        self.p = p
        self.cellmeasure = mesh.cell_area()
        self.dof = LinearMeshCFEDof(mesh, p)
        self.multi_index_matrix = mesh.multi_index_matrix

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
        return self.dof.interpolation_points()

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.cell2dof[index]

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof()

    def edge_to_dof(self, index: Index=_S):
        return self.dof.edge_to_dof()

    def is_boundary_dof(self, threshold=None) -> TensorLike:
        return self.dof.is_boundary_dof(threshold=threshold)

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD
    
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

        Notes
        -----
        计算基函数在重心坐标点处的函数值，注意 bc 的形状为 (..., TD+1), TD 为 bc
        所在空间的拓扑维数。

        """
        
        p = self.p
        phi = self.mesh.shape_function(bc, p=p)
        return phi[..., None, :] 

    @barycentric
    def grad_basis(self, bc: TensorLike, index: Index=_S, variables='x'):
        """

        Notes
        -----
        计算空间基函数关于实际坐标点 x 的梯度。
        """
        p = self.p
        gphi = self.mesh.grad_shape_function(bc, index=index, p=p,
                variables=variables)
        return gphi

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = bm.einsum(s1, phi, uh[cell2dof])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index:Index=_S):
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell2dof[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = bm.einsum(s1, gphi, uh[cell2dof])
        return val

    def integral_basis(self, q=None):
        cell2dof = self.cell_to_dof()
        qf = self.quadrature_formula if q is None else self.mesh.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        rm = self.mesh.reference_cell_measure()
        G = self.mesh.first_fundamental_form(bcs)
        d = bm.sqrt(bm.linalg.det(G))
        phi = self.basis(bcs)
        cc = bm.einsum('q, qci, qc->ci', ws*rm, phi, d)
        gdof = self.number_of_global_dofs()
        c = bm.zeros(gdof, dtype=self.ftype)
        bm.add.at(c, cell2dof, cc)
        return c


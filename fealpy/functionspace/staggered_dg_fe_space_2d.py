from typing import Union, TypeVar, Generic, Callable, Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from fealpy.decorator import barycentric


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

class StaggeredDGFESpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int, m: int, device=None):
        self.mesh = mesh
        self.p = p
        self.m = m

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device
        self.ikwargs = bm.context(mesh.cell)
        self.fkwargs = bm.context(mesh.node)

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()


    def number_of_local_dofs(self, etype) -> int: 
        pass
    def number_of_global_dofs(self) -> int:
        pass
    def cell_to_dof(self, index=_S):
        pass
    def is_boundary_dof(self, threshold, method="interp"): 
        pass
    def basis(self, bcs, index=_S):
        pass
    def boundary_interpolate(self, gd, uh, threshold=None, method="interp"):
        pass
    def interpolation(self, flist):
        pass

    @barycentric
    def value(self, uh ,bc, index=_S):
        pass
    @barycentric
    def grad_value(self, uh, bcs):
        return self.grad_m_value(uh, bcs, 1)


  

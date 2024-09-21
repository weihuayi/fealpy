from typing import Union, TypeVar, Generic, Callable, Optional 
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from fealpy.experimental.functionspace.bernstein_fe_space import BernsteinFESpace
from .functional import symmetry_span_array, symmetry_index, span_array
from scipy.special import factorial, comb
from fealpy.decorator import barycentric

_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

class CmConformingFESpace3d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh:_MT, p: int, m: int, isCornerNode:bool):
        assert(p>8*m)
        self.mesh = mesh
        self.p = p
        self.m = m
        self.isCornerNode = isCornerNode
        self.bspace = BernsteinFESpace(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        #self.device = mesh.device

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        #self.coeff = self.coefficient_matrix()
        self.multiIndex = mesh.multi_index_matrix(p, 3)
    
    def get_dof_index(self):
        p = self.p
        m = self.m
        midx = self.multiIndex
        ldof = midx.shape[0]

        isn_cell_dof = bm.zeros(ldof, dtype=bm.bool)

        node_dof_index = []
        for i in range(4):
            a = []
            for r in range(4*m+1):
                Dvr = midx[:, i]==p-r
                isn_cell_dof = isn_cell_dof | Dvr
                a.append(bm.where(bm.array(Dvr))[0])
            node_dof_index.append(a)

        locEdge = bm.array([])
        face_dof_index = []
        for i in range(6):
            a = []
            for r in range(2*m+1):
                aa = []
                for j in range(r+1):
                    Derj = (bm.sum(midx[:, locEdge[i]], axis=1)==p-r) & (midx[:, dualEdge[i, 1]]==j) & (~isn_cell_dof)
                    isn_cell_dof = isn_cell_dof | Derj


        return

    def number_of_local_dofs(self, etype, p=None) -> int:
        p = self.p if p is None else p
        m = self.m
        if etype=="cell":
            return (p+1)*(p+2)*(p+3)//6




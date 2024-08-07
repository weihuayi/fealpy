from typing import Union, TypeVar, Generic, Callable, Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace

_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

class CmConformingFESpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int, m: int, isCornerNode: bool):
        assert(p>4*m)
        self.mesh = mesh
        self.p = p
        self.m = m
        self.isCornerNode = isCornerNode

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def number_of_local_dofs(self, etype) -> int: 
        # TODO:这个用到过吗
        p = self.p
        m = self.m
        if etype=="cell":
            return (p+1)*(p+1)//2
        if etype=="edge":
            ndof = (2*m+1)*(2*m+2)//2
            eidof = (m+1)*(2*p-7*m-2)//2
            return ndof+eidof 
        if etype=="node":
            return (2*m+1)*(2*m+2)//2
    def number_of_internal_dofs(self, etype) -> int:
        p = self.p
        m = self.m
        if etype=="node":
            return (2*m+1)*(2*m+2)//2
        if etype=="edge":
            return (m+1)*(2*p-7*m-2)//2
        if etype=="cell":
            ndof = (2*m+1)*(2*m+2)//2
            eidof = (m+1)*(2*p-7*m-2)//2
            return (p+1)*(p+2)//2 - (ndof+eidof)*3 

    def number_of_global_dofs(self) -> int:
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        ndof = self.number_of_internal_dofs("node") 
        eidof = self.number_of_internal_dofs("edge") 
        cidof = self.number_of_internal_dofs("cell") 
        return NN*ndof + NE*eidof + NC* cidof

    def node_to_dof(self):
        m = self.m
        mesh = self.mesh
        ndof = self.number_of_internal_dofs('node')

        NN = mesh.number_of_nodes()
        n2d = bm.arange(NN*ndof, dtype=self.itype).reshape(NN, ndof)
        return n2d

    def edge_to_internal_dof(self):
        p = self.p
        m = self.m
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge')
        mesh = self.mesh 
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        e2id = bm.arange(NN*ndof, NN*ndof+NE*eidof,dtype=self.itype).reshape(NE, eidof)
        return e2id
    def cell_to_internal_dof(self):
        p = self.p
        m = self.m
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge')
        cidof = self.number_of_internal_dofs('cell')
        mesh = self.mesh 
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        c2id = bm.arange(NN*ndof + NE*eidof,
                NN*ndof+NE*eidof+NC*cidof,dtype=self.itype).reshape(NC, cidof)
        return c2id
#    def cell_to_dof(self):



from typing import Union, TypeVar, Generic, Callable,Optional

from ..backend import TensorLike
from ..backend import backend_manager as bm
from .space import FunctionSpace
from fealpy.functionspace import BernsteinFESpace  # 未实现
from fealpy.quadrature import FEMeshIntegralAlg    # 未实现
from fealpy.functionspace.femdof import multi_index_matrix2d  # 未实现

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.decorator import cartesian, barycentric  # 未实现

Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

# 自由度管理
class FirstNedelecDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.multiindex = multi_index_matrix2d(p)

    def number_of_local_dofs(self,doftype ='all'):
        p = self.p
        if doftype == 'all':
            return (p+1)*(p+3)
        elif doftype in {'cell', 2}:
            return (p+1)*p
        elif doftype in {'face','edge', 1}:
            return p+1
        elif doftype in {'node',0}:
            return 0

    def number_of_global_dof(self):
        NC =  self.mesh.number_of_cells()
        NE =  self.mesh.number_of_edges()
        edof = self.number_of_local_dofs("edge")
        cdof = self.number_of_local_dofs("cell")
        return NE*edof + NC*cdof

    def edge_to_dof(self,index = _S):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs("edge") 
        return bm.arange(NE*edof).reshape(NE,edof)[index]

    def cell_to_dof(self):
        p = self.p
        cdof = self.number_of_local_dofs("cell")
        edof = self.number_of_local_dofs("edge")
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dof()
        e2dof = self.edge_to_dof()

        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        c2e = self.mesh.cell_to_edge()
        edge = self.mesh.entity("edge")
        cell = self.mesh.entity("cell")

        c2d = bm.zeros((NC,ldof),dtype= bm.int_)
        c2d[:,:edof*3] = e2dof[c2e].reshape(NC,edof*3)
        s = [1, 0, 0]
        for i in range(3):
            flag = cell[:,s[i]] == edge[c2e[:,i],1]
            c2d[flag,edof*i:edof*(i+1)] == c2d[flag,edof*i:edof*(i+1)][:,::-1]
        c2d[:,edof*3:] = bm.arange(NE*edof,gdof).reshape(NC,-1)
        return c2d
    
    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def boundary_dof(self):
        eidx = self.mesh.boundary_edge_index()
        e2d = self.edge_to_dof(index=eidx)
        return e2d.reshape(-1)

    def is_boundary_dof(self):
        bddof = self.boundary_dof()

        gdof = self.number_of_global_dofs()
        flag = bm.zeros(gdof, dtype=bm.bool_)

        flag[bddof] = True
        return flag


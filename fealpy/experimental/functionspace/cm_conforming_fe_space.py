from typing import Union, TypeVar, Generic, Callable, Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from scipy.special import factorial, comb

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
            return (p+1)*(p+2)//2
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
    def cell_to_dof(self):
        p = self.p
        m = self.m
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        cell = mesh.entity('cell')
        edge = mesh.entity('edge')
        NC = mesh.number_of_cells()
        cell2edge = mesh.cell_to_edge()
        ldof = self.number_of_local_dofs('cell')
        cidof = self.number_of_internal_dofs('cell')
        eidof = self.number_of_internal_dofs('edge')
        ndof = self.number_of_internal_dofs('node')
        c2d = bm.zeros((NC, ldof), dtype=self.itype)
        e2id = self.edge_to_internal_dof()
        n2d = self.node_to_dof()
        c2d[:, ldof-cidof:] = bm.arange(NN*ndof + NE*eidof,
                NN*ndof+NE*eidof+NC*cidof,dtype=self.itype).reshape(NC, cidof)
        c2d[:, :ndof*3] = n2d[cell].reshape(NC,-1) 
        c2eSign = mesh.cell_to_edge_sign()
        for e in bm.arange(3):
            N = ndof*3+eidof*e
            flag = ~c2eSign[:, e]
            c2d[:, N:N+eidof] = e2id[cell2edge[:, e]]
            n0, n1 = 0, p-4*m-1
            for l in bm.arange(m+1):
                c2d[flag, N+n0:N+n0+n1] = bm.flip(c2d[flag,
                    N+n0:N+n0+n1],axis=1)
                n0 += n1
                n1 += 1
        return c2d
    def is_boundary_dof(self):
        p = self.p
        m = self.m
        gdof = self.number_of_global_dofs()
        isBdDof = bm.zeros(gdof, dtype=bm.bool)
        isBdEdge = self.mesh.boundary_face_flag() #TODO:ds 中没有edge
        isBdNode = self.mesh.boundary_node_flag()
        # 边界边
        bedof = self.edge_to_internal_dof()[isBdEdge]
        isBdDof[bedof] = True
        # 角点所有
        bndof = self.node_to_dof()
        isBdDof[bndof[self.isCornerNode]] = True
        # 非角点边界点
        bndof = bndof[isBdNode]
        k = 0
        for r in range(2*m+1):
            L = min(m+1, r+1)
            isBdDof[bndof[:, k:k+L]] = True
            k += r+1
        return isBdDof

    def coefficient_matrix(self):
        p = self.p
        m = self.m
        mesh = self.mesh

        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs('cell')
        tem = bm.eye(ldof, dtype=self.ftype)
        coeff = bm.tile(tem, (NC, 1, 1))

        # 多重指标
        multiIndex = mesh.multi_index_matrix(p, 2)
        S02m0 = multiIndex[multiIndex[:, 0]>=p-2*m]
        S02m1 = S02m0[:, [1, 0, 2]]
        S02m2 = S02m0[:, [1, 2, 0]]
        S1m0 = multiIndex[(multiIndex[:, 0]<=m) & (bm.all(multiIndex[:, 1:]<p-2*m,
                axis=1))]
        S1m0 = S1m0[:, [0,2,1]][::-1]
        S1m1 = S1m0[:, [2, 0, 1]]
        S1m2 = S1m0[:, [1, 2, 0]]
        S2 = multiIndex[bm.all(multiIndex>m, axis=1)]
        dof2midx = bm.concatenate([S02m0, S02m1, S02m2, S1m0, S1m1, S1m2, S2])  
        midx2num = lambda a: (a[:, 1]+a[:, 2])*(1+a[:, 1]+ a[:, 2])//2 + a[:, 2]
        dof2num = midx2num(dof2midx)
        dof2num = np.agrsort(dof2num)

        # 节点局部自由度
        S02m = [S02m0, S02m1, S02m2]
        for v in range(3):
            flag = bm.ones(3, dtype=bm.bool)
            flag[v] = False
            for alpha in S02m[v]:
                i = midx2num(alpha[None, :])
                i = dof2num[i]
                alpha = alpha[flag]
                r = bm.sum(alpha)
                betas = multiIndex(r, 2)
                for beta in betas:
                    if bm.all(alpha-beta[v]<=0):
                        continue
                    j = bm.sum(beta) + r
                    j = midx2num(j)
                    j = dof2num[j]
                    coeff[:, i, j] = (-1)**(beta[v])*comb(p, r) 



                



        return

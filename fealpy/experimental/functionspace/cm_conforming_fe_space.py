from typing import Union, TypeVar, Generic, Callable, Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .functional import symmetry_span_array, symmetry_index
from scipy.special import factorial, comb
from scipy.linalg import solve_triangular


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
        isCornerNode = self.isCornerNode

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
        S1m0 = bm.flip(S1m0[:, [0,2,1]], axis=0)
        S1m1 = S1m0[:, [2, 0, 1]]
        S1m2 = S1m0[:, [1, 2, 0]]
        S2 = multiIndex[bm.all(multiIndex>m, axis=1)]
        dof2midx = bm.concatenate([S02m0, S02m1, S02m2, S1m0, S1m1, S1m2, S2])  
        midx2num = lambda a: (a[:, 1]+a[:, 2])*(1+a[:, 1]+ a[:, 2])//2 + a[:, 2]
        dof2num = midx2num(dof2midx)
        dof2num = bm.argsort(dof2num)

        # 局部自由度 顶点
        S02m = [S02m0, S02m1, S02m2]
        for v in range(3):
            flag = bm.ones(3, dtype=bm.bool)
            flag[v] = False
            for alpha in S02m[v]:
                i = midx2num(alpha[None, :])
                i = dof2num[i]
                dalpha = alpha[flag]
                r = int(bm.sum(dalpha))
                betas = mesh.multi_index_matrix(r, 2)
                for beta in betas:
                    if bm.all(alpha-beta[v]<=0):
                        continue
                    sign = (-1)**(beta[v])
                    beta[v] = beta[v] + alpha[v]
                    j = midx2num(beta[None, :])
                    j = dof2num[j]
                    dbeta = beta[flag]
                    coeff[:, i, j] = sign*comb(bm.sum(beta),
                            r)*bm.prod(bm.array(comb(dalpha, dbeta)),
                                    axis=0)*factorial(r) 


        # 局部自由度 边
        glambda = mesh.grad_lambda()
        c2e = mesh.cell_to_edge()
        n = mesh.edge_normal()[c2e]
        N = bm.einsum('cfd, ced->cef', glambda, n)
        S1m = [S1m0, S1m1, S1m2]
        for de in range(3):
            e = bm.ones(3, dtype=bm.bool)
            e[de] = False # e
            for alpha in S1m[de]:
                i = midx2num(alpha[None, :])
                i = dof2num[i]
                dalpha = alpha[de] 
                betas = mesh.multi_index_matrix(int(dalpha), 2)
                for beta in betas:
                    val = bm.prod(N[:, de]**beta,axis=1)/bm.prod(bm.array(factorial(beta)), axis=0)
                    beta[e] = beta[e] +alpha[e]
                    j = midx2num(beta[None, :])
                    j = dof2num[j]
                    coeff[:, i, j] = comb(bm.sum(beta),
                                          int(dalpha))*(factorial(int(dalpha)))**2*val[:, None]
        #for i in range(NC):
        
        #    coeff[i] = solve_triangular(coeff[i], bm.eye(ldof), lower=True)
        coeff = bm.linalg.inv(coeff)

        # 全局自由度
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        Ndelat = bm.zeros((NC, 3, 2, 2), dtype=self.ftype)
        t0 = node[cell[:, 2]] - node[cell[:, 1]]
        t1 = node[cell[:, 0]] - node[cell[:, 2]]
        t2 = node[cell[:, 1]] - node[cell[:, 0]]
        Ndelat[:, 0, 0] = t2
        Ndelat[:, 0, 1] = -t1
        Ndelat[:, 1, 0] = -t2
        Ndelat[:, 1, 1] = t0
        Ndelat[:, 2, 0] = t1
        Ndelat[:, 2, 1] = -t0
        ndof = self.number_of_internal_dofs('node')
        coeff1 = bm.zeros((NC, 3*ndof, 3*ndof), dtype=self.ftype)
        symidx = [symmetry_index(2, r) for r in range(1, 2*m+1)]
        # 边界自由度
        isBdNode = mesh.boundary_node_flag()
        isBdEdge = mesh.boundary_face_flag()
        edge = mesh.entity('edge')[isBdEdge]
        n = mesh.edge_unit_normal()[isBdEdge]
        NN = isBdNode.sum()
        coeff2 = bm.tile(bm.eye(3*ndof, dtype=self.ftype), (NC, 1, 1))
        nodefram = bm.zeros((NN, 2, 2), dtype=self.ftype)
        bdnidxmap = bm.zeros(len(isBdNode),dtype=self.itype) 
        bdnidxmap[isBdNode] = bm.arange(NN, dtype=self.itype)
        nodefram[bdnidxmap[edge[:, 0]], 1] += 0.5*n
        nodefram[bdnidxmap[edge[:, 1]], 1] += 0.5*n
        nodefram[:, 0] = nodefram[:, 1]@bm.array([[0,1],[-1, 0]],
                                                 dtype=self.ftype)
        kk = 0
        for v in range(3):
            flag = bm.ones(3, dtype=bm.bool) 
            flag[v] = False #v^*
            coeff1[:, ndof*v, ndof*v] = 1
            cidx = isBdNode[cell[:, v]] & ~isCornerNode[cell[:, v]] 
            for gamma in S02m[v][1:]:
                i = midx2num(gamma[None, :])
                i = dof2num[i]
                i = int(i)
                gamma = gamma[flag]
                Ndelta_sym = symmetry_span_array(Ndelat[:, v],
                                                 gamma).reshape(NC, -1)
                r = int(bm.sum(gamma))
                j = (r+1)*(r+2)//2
                c = symidx[r-1][1]
                coeff1[:, i, j-r-1+kk:j+kk] = Ndelta_sym[:, symidx[r-1][0]] *c[None, :] 
                if bm.sum(cidx)>0:
                    tn_sym = symmetry_span_array(nodefram[bdnidxmap[cell[cidx][:, v]]], gamma).reshape(bm.sum(cidx), -1)
                    coeff2[cidx, i, j-r-1+kk:j+kk] = tn_sym[:, symidx[r-1][0]]*c[None, :]
            kk = kk+ndof

        coeff1 = bm.einsum('cik,cjk->cij', coeff1, coeff2)
        coeff = bm.transpose(coeff, (0, 2, 1))
        coeff[:, :3*ndof] = bm.einsum('cji,cjk->cik', coeff1, coeff[:, :3*ndof])
        return coeff[:, :, dof2num]





                













from typing import Union, TypeVar, Generic, Callable, Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .bernstein_fe_space import BernsteinFESpace
from .functional import symmetry_span_array, symmetry_index, span_array
from scipy.special import factorial, comb
from scipy.linalg import solve_triangular
from fealpy.decorator import barycentric


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

class CmConformingFESpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int, m: int, device=None):
        assert(p>4*m)
        self.mesh = mesh
        self.p = p
        self.m = m
        self.isCornerNode = self.isCornerNode()
        self.bspace = BernsteinFESpace(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device
        self.ikwargs = bm.context(cell)
        self.fkwargs = bm.context(node)

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.coeff = self.coefficient_matrix()
    def isCornerNode(self):
        mesh = self.mesh
        edge = mesh.entity('edge')
        NN = mesh.number_of_nodes()
        boundary_edge = mesh.boundary_edge_flag()
        edge = edge[boundary_edge]
        en = mesh.edge_unit_normal()[boundary_edge]

        nnn = bm.zeros((NN,2,2))
        isCornerNode = bm.zeros(NN, dtype=bm.bool, device=self.device)
        nnn[edge[:, 0],0] = en
        nnn[edge[:, 1],1] = en
        
        flag = bm.abs(bm.cross(nnn[:,0,:],nnn[:,1,:],axis=1))>1e-10
        return flag


    def number_of_local_dofs(self, etype) -> int: 
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
        n2d = bm.arange(NN*ndof, dtype=self.itype, device=self.device).reshape(NN, ndof)
        return n2d

    def edge_to_internal_dof(self):
        p = self.p
        m = self.m
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge')
        mesh = self.mesh 
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        e2id = bm.arange(NN*ndof, NN*ndof+NE*eidof,dtype=self.itype, device=self.device).reshape(NE, eidof)
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
                NN*ndof+NE*eidof+NC*cidof,dtype=self.itype, device=self.device).reshape(NC, cidof)
        return c2id
    def cell_to_dof(self, index=_S):
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
        c2d = bm.zeros((NC, ldof), dtype=self.itype, device=self.device)
        e2id = self.edge_to_internal_dof()
        n2d = self.node_to_dof()
        c2d[:, ldof-cidof:] = bm.arange(NN*ndof + NE*eidof,
                NN*ndof+NE*eidof+NC*cidof,dtype=self.itype, device=self.device).reshape(NC, cidof)
        c2d[:, :ndof*3] = n2d[cell].reshape(NC,-1) 
        c2eSign = mesh.cell_to_face_sign()
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
        return c2d[index]
    def is_boundary_dof(self, threshold, method="interp"): #TODO:这个threshold 没有实现
        p = self.p
        m = self.m
        gdof = self.number_of_global_dofs()
        isBdDof = bm.zeros(gdof, dtype=bm.bool, device=self.device)
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
        device = bm.get_device(mesh.cell)
        isCornerNode = self.isCornerNode

        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs('cell')
        tem = bm.eye(ldof, dtype=self.ftype, device=self.device)
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
        dof2num = bm.argsort(dof2num, axis=0)

        # 局部自由度 顶点
        S02m = [S02m0, S02m1, S02m2]
        for v in range(3):
            flag = bm.ones(3, dtype=bm.bool, device=self.device)
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
                    beta_cpu = bm.to_numpy(bm.sum(beta))
                    dalpha_cpu = bm.to_numpy(dalpha)
                    dbeta_cpu = bm.to_numpy(dbeta)
                    coeff[:, i, j] = sign*comb(beta_cpu,r)*bm.prod(bm.array(
                        comb(dalpha_cpu, dbeta_cpu),device=device),axis=0)*factorial(r)
                    #coeff[:, i, j] = sign*comb(bm.sum(beta),r)*bm.prod(bm.array(comb(dalpha, dbeta),**ikwargs),
                    #                                                   axis=0)*factorial(r)

        #import numpy as np
        #for i  in range(NC-1):
        #    np.testing.assert_allclose(coeff[i],coeff[i+1], atol=1e-15)
        #    print(i)


        # 局部自由度 边
        glambda = mesh.grad_lambda()
        c2e = mesh.cell_to_edge()
        n = mesh.edge_normal()[c2e]
        N = bm.einsum('cfd, ced->cef', glambda, n)
        S1m = [S1m0, S1m1, S1m2]
        for de in range(3):
            e = bm.ones(3, dtype=bm.bool, device=self.device)
            e[de] = False # e
            for alpha in S1m[de]:
                i = midx2num(alpha[None, :])
                i = dof2num[i]
                dalpha = alpha[de] 
                betas = mesh.multi_index_matrix(int(dalpha), 2)
                for beta in betas:
                    beta_cpu = bm.to_numpy(beta)
                    val = bm.prod(N[:, de]**beta,axis=1)/bm.prod(bm.array(factorial(beta_cpu), device=device), axis=0)
                    beta[e] = beta[e] +alpha[e]
                    beta_sum_cpu = bm.to_numpy(bm.sum(beta))
                    j = midx2num(beta[None, :])
                    j = dof2num[j]
                    coeff[:, i, j] = comb(beta_sum_cpu,
                                          int(dalpha))*(factorial(int(dalpha)))**2*val[:, None]
        
        ## 测试每个单元顶点的系数矩阵是一样的,与单元无关
        #import numpy as np
        #for i  in range(NC-1):
        #    np.testing.assert_allclose(coeff[i,:18,:18],coeff[i+1,:18,:18], atol=1e-15)
        #    print(i)
        #print(coeff[:, :18, :18])
        #import numpy as np
        #np.savetxt('c.csv', coeff[0], delimiter=',')
        

        #for i in range(NC):
        
        #    coeff[i] = solve_triangular(coeff[i], bm.eye(ldof), lower=True)
        coeff = bm.linalg.inv(coeff)

        # 全局自由度
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        Ndelat = bm.zeros((NC, 3, 2, 2), dtype=self.ftype, device=self.device)
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
        coeff1 = bm.zeros((NC, 3*ndof, 3*ndof), dtype=self.ftype, device=self.device)
        symidx = [symmetry_index(2, r, device=device) for r in range(1, 2*m+1)]
        # 边界自由度
        isBdNode = mesh.boundary_node_flag()
        isBdEdge = mesh.boundary_face_flag()
        edge = mesh.entity('edge')[isBdEdge]
        n = mesh.edge_unit_normal()[isBdEdge]
        NN = isBdNode.sum()
        coeff2 = bm.tile(bm.eye(3*ndof, dtype=self.ftype, device=self.device), (NC, 1, 1))
        nodefram = bm.zeros((NN, 2, 2), dtype=self.ftype, device=self.device)
        bdnidxmap = bm.zeros(len(isBdNode),dtype=self.itype, device=self.device) 
        bdnidxmap[isBdNode] = bm.arange(NN, dtype=self.itype, device=self.device)
        nodefram[bdnidxmap[edge[:, 0]], 1] += 0.5*n
        nodefram[bdnidxmap[edge[:, 1]], 1] += 0.5*n
        nodefram[:, 0] = nodefram[:, 1]@bm.array([[0,1],[-1, 0]],
                                                 dtype=self.ftype, device=self.device)
        kk = 0
        for v in range(3):
            flag = bm.ones(3, dtype=bm.bool, device=self.device) 
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

    def basis(self, bcs, index=_S):#TODO:这个index没实现
        coeff = self.coeff
        bphi = self.bspace.basis(bcs)
        return bm.einsum('cil, cql->cqi', coeff, bphi)

    def grad_m_basis(self, bcs, m):
        coeff = self.coeff
        bgmphi = self.bspace.grad_m_basis(bcs, m)
        return bm.einsum('cil, cqlg->cqig', coeff, bgmphi)


    def boundary_interpolate(self, gd, uh, threshold=None, method="interp"):
        '''
        @param gd : [right, tr], 第一个位置为方程的右端项，第二个位置为迹函数
        '''
        #TODO 只处理的边界为 0 的情况
        mesh = self.mesh
        m = self.m
        p = self.p
        isCornerNode = self.isCornerNode 
        coridx = bm.where(isCornerNode)[0]
        isBdNode = mesh.boundary_node_flag()
        isBdEdge = mesh.boundary_face_flag()
        bdnidxmap = bm.zeros(len(isBdNode), dtype=bm.int32, device=self.device)
        bdnidxmap[isBdNode] = bm.arange(isBdNode.sum(),dtype=bm.int32, device=self.device)
        n2id = self.node_to_dof()[isBdNode]
        e2id = self.edge_to_internal_dof()[isBdEdge]

        node = self.mesh.entity('node')[isBdNode]
        edge = self.mesh.entity('edge')[isBdEdge]
        NN = len(node)
        NE = len(edge)

        n = mesh.edge_unit_normal()[isBdEdge]
        nodefram = bm.zeros((NN, 2, 2), dtype=bm.float64, device=self.device) ## 注意！！！先 t 后 n
        nodefram[bdnidxmap[edge[:, 0]], 1] += 0.5*n
        nodefram[bdnidxmap[edge[:, 1]], 1] += 0.5*n
        nodefram[:, 0] = nodefram[:, 1]@bm.array([[0, 1], [-1,
                                                           0]],dtype=bm.float64, device=self.device)
        nodefram[bdnidxmap[coridx]] = bm.tile(bm.eye(2,dtype=bm.float64, device=self.device), (len(coridx), 1, 1))

        # 顶点自由度
        uh[n2id[:, 0]] = gd[0](node) 
        k = 1; 
        for r in range(1, 2*m+1):
            val = gd[r](node) 
            multiIndex = self.mesh.multi_index_matrix(r, 1)
            symidx, num = symmetry_index(2, r, device=self.device)

            #idx = self.mesh.multi_index_matrix(r, 1)
            #num = factorial(r)/bm.prod(factorial(idx), axis=1)

            L = min(m+1, r+1)
            for j in range(L):
                nt = symmetry_span_array(nodefram, multiIndex[j])
                uh[n2id[:, k+j]] = bm.einsum("ni, ni, i->n", val, nt.reshape(NN, -1)[:, symidx], num)

            for j in range(L, r+1):
                idx = bdnidxmap[coridx]
                nt = symmetry_span_array(nodefram[idx], multiIndex[j])
                uh[n2id[idx, k+j]] = bm.einsum("ni, ni, i->n", val[idx], nt.reshape(idx.shape[0], -1)[:, symidx], num)
            k += r+1

        # 边上的自由度
        n = mesh.edge_normal()[isBdEdge]
        k = 0; l = p-4*m-1
        for r in range(m+1):
            bcs = self.mesh.multi_index_matrix(p-r, 1,dtype=self.ftype)/(p-r)
            b2l = self.bspace.bernstein_to_lagrange(p-r, 1)
            point = self.mesh.bc_to_point(bcs)[isBdEdge]
            if r==0:
                ffval = gd[0](point) #(ldof, NE)
                bcoeff = bm.einsum('el, il->ei', ffval, b2l)
                uh[e2id[:, k:l]] = bcoeff[:, 2*m+1-r: -2*m-1+r]
            else:
                symidx, num = symmetry_index(2, r, device=self.device)
                nnn = span_array(n[:, None, :], bm.array([r]))
                nnn = nnn.reshape(NE, -1)[:, symidx]

                #GD = self.mesh.geo_dimension()
                #idx = self.mesh.multi_index_matrix(r, GD-1)
                #num = factorial(r)/bm.prod(factorial(idx), axis=1)

                ffval = gd[r](point) #(ldof, NE, L), L 指的是分量个数，k 阶导有 k 个
                bcoeff = bm.einsum('ej, elj, j, il->ei', nnn, ffval, num, b2l)
                uh[e2id[:, k:l]] = bcoeff[:, 2*m+1-r: -2*m-1+r]
            k = l
            l += p-4*m+r
        isDDof = self.is_boundary_dof(threshold=threshold)
        uI = self.interpolation(gd)
        isDDofidx = bm.where(isDDof)[0]
        #uh[isDDof] = uI[isDDof]
        return uh, isDDof

    def interpolation(self, flist):
        """
        @breif 对函数进行插值，其中 flist 是一个长度为 2m+1 的列表，flist[k] 是 f 的 k
            阶导组成的列表, 如 m = 1 时，flist = [[f], [fx, fy], [fxx, fxy, fyy]]
        """

        m = self.m
        p = self.p
        mesh = self.mesh
        fI = self.function()

        n2id = self.node_to_dof()
        e2id = self.edge_to_internal_dof()
        c2id = self.cell_to_internal_dof()

        node = self.mesh.entity('node')

        # 顶点自由度
        fI[n2id[:, 0]] = flist[0](node) 
        k = 1; 
        for r in range(1, 2*m+1):
            fI[n2id[:, k:k+r+1]] = flist[r](node) #这里不用乘标架吗? 
            k += r+1

        # 边上的自由度
        NE = mesh.number_of_edges()
        n = mesh.edge_normal()
        k = 0; l = p-4*m-1
        for r in range(m+1):
            bcs = self.mesh.multi_index_matrix(p-r, 1,dtype=self.ftype)/(p-r)
            b2l = self.bspace.bernstein_to_lagrange(p-r, 1)
            point = self.mesh.bc_to_point(bcs)
            if r==0:
                ffval = flist[0](point) #(ldof, NE)
                bcoeff = bm.einsum('el, il->ei', ffval, b2l)
                fI[e2id[:, k:l]] = bcoeff[:, 2*m+1-r: -2*m-1+r]
            else:
                symidx, num = symmetry_index(2, r, device=self.device)

                nnn = span_array(n[:, None, :], bm.array([r], device=self.device))
                nnn = nnn.reshape(NE, -1)[:, symidx]

                #GD = self.mesh.geo_dimension()
                #idx = self.mesh.multi_index_matrix(r, GD-1)
                #num = factorial(r)/bm.prod(factorial(idx), axis=1)

                ffval = flist[r](point) #(ldof, NE, L), L 指的是分量个数，k 阶导有 k 个
                bcoeff = bm.einsum('ej, elj, j, il->ei', nnn, ffval, num, b2l)
                fI[e2id[:, k:l]] = bcoeff[:, 2*m+1-r: -2*m-1+r]
            k = l
            l += p-4*m+r

        #内部自由度
        midx = self.mesh.multi_index_matrix(p, 2, dtype=self.itype)
        bcs = self.mesh.multi_index_matrix(p, 2,dtype=self.ftype)/p
        b2l = self.bspace.bernstein_to_lagrange(p, 2)
        point = self.mesh.bc_to_point(bcs)
        ffval = flist[0](point) #(ldof, NE)
        bcoeff = bm.einsum('el, il->ei', ffval, b2l)

        flag = bm.all(midx > m, axis=1)# & bm.all(midx < p-2*m-1, axis=1)
        fI[c2id] = bcoeff[:, flag]
    
        # 边界节点上的自由度的处理
        isCornerNode = self.isCornerNode 
        coridx = bm.where(isCornerNode)[0]
        isBdNode = mesh.boundary_node_flag()
        isBdEdge = mesh.boundary_face_flag()
        bdnidxmap = bm.zeros(len(isBdNode), dtype=bm.int32, device=self.device)
        bdnidxmap[isBdNode] = bm.arange(isBdNode.sum(),dtype=bm.int32, device=self.device)
        n2id = self.node_to_dof()[isBdNode]
        e2id = self.edge_to_internal_dof()[isBdEdge]

        node = self.mesh.entity('node')[isBdNode]
        edge = self.mesh.entity('edge')[isBdEdge]
        NN = len(node)
        NE = len(edge)

        n = mesh.edge_unit_normal()[isBdEdge]
        nodefram = bm.zeros((NN, 2, 2), dtype=bm.float64, device=self.device) ## 注意！！！先 t 后 n
        nodefram[bdnidxmap[edge[:, 0]], 1] += 0.5*n
        nodefram[bdnidxmap[edge[:, 1]], 1] += 0.5*n
        nodefram[:, 0] = nodefram[:, 1]@bm.array([[0, 1], [-1,
                                                           0]],dtype=bm.float64, device=self.device)
        nodefram[bdnidxmap[coridx]] = bm.tile(bm.eye(2,dtype=self.ftype, device=self.device), (len(coridx), 1, 1))

        k = 1; 
        for r in range(1, 2*m+1):
            val = flist[r](node) 
            multiIndex = self.mesh.multi_index_matrix(r, 1)
            symidx, num = symmetry_index(2, r, device=self.device)

            #idx = self.mesh.multi_index_matrix(r, 1)
            #num = factorial(r)/bm.prod(factorial(idx), axis=1)

            L = min(m+1, r+1)
            for j in range(r+1):
                nt = symmetry_span_array(nodefram, multiIndex[j])
                fI[n2id[:, k+j]] = bm.einsum("ni, ni, i->n", val, nt.reshape(NN, -1)[:, symidx], num)
            k += r+1
        #self.boundary_interpolate(flist, fI)
        return fI

    @barycentric
    def value(self, uh ,bc, index=_S):
        """
        @brief Computes the value of a finite element function `uh` at a set of
        barycentric coordinates `bc` for each mesh cell.

        @param uh: numpy.ndarray, the dof coefficients of the basis functions.
        @param bc: numpy.ndarray, the barycentric coordinates with shape (NQ, TD+1).
        @param index: Union[numpy.ndarray, slice], index of the entities (default: np.s_[:]).
        @return numpy.ndarray, the computed function values.

        This function takes the dof coefficients of the finite element function `uh` and a set of barycentric
        coordinates `bc` for each mesh cell. It computes the function values at these coordinates
        and returns the results as a numpy.ndarray.
        """
        phi = self.basis(bc) # (NQ, 1, ldof)
        cell2dof = self.cell_to_dof()
        val = bm.einsum('cql, cl->cq', phi, uh[cell2dof])
        return val

    @barycentric
    def grad_m_value(self, uh, bcs, m):
        gmphi = self.grad_m_basis(bcs, m) # (NQ, 1, ldof)
        cell2dof = self.cell_to_dof()
        val = bm.einsum('cqlg, cl->cqg', gmphi, uh[cell2dof])
        return val

    @barycentric
    def grad_value(self, uh, bcs):
        return self.grad_m_value(uh, bcs, 1)




                













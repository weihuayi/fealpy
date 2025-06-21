import numpy as np
import time
import string
from ..decorator import barycentric
from .fem_dofs import *
from .Function import Function
from ..common.tensor import *

from fealpy.old.quadrature.TetrahedronQuadrature import TetrahedronQuadrature
from scipy.special import factorial, comb
from scipy.sparse import csc_matrix, csr_matrix

class BernsteinFESpace:
    DOF = { 'C': {
                "IntervalMesh": IntervalMeshCFEDof,
                "TriangleMesh": TriangleMeshCFEDof,
                "TetrahedronMesh": TetrahedronMeshCFEDof,
                "EdgeMesh": EdgeMeshCFEDof,
                }, 
            'D':{
                "IntervalMesh": IntervalMeshDFEDof,
                "TriangleMesh": TriangleMeshDFEDof,
                "TetrahedronMesh": TetrahedronMeshDFEDof,
                "EdgeMesh": EdgeMeshDFEDof, 
                }
        } 

    def __init__(self, 
            mesh, 
            p: int=1, 
            spacetype: str='C', 
            doforder: str='vdims'):
        """
        @brief Initialize the Lagrange finite element space.

        @param mesh The mesh object.
        @param p The order of interpolation polynomial, default is 1.
        @param spacetype The space type, either 'C' or 'D'.
        @param doforder The convention for ordering degrees of freedom in vector space, either 'sdofs' (default) or 'vdims'.

        @note 'sdofs': 标量自由度优先排序，例如 x_0, x_1, ..., y_0, y_1, ..., z_0, z_1, ...
              'vdims': 向量分量优先排序，例如 x_0, y_0, z_0, x_1, y_1, z_1, ...
        """
        self.mesh = mesh
        self.p = p
        assert spacetype in {'C', 'D'} 
        self.spacetype = spacetype
        self.btype = "BS"   # 基函数类型 BS, BR, BC
        self.doforder = doforder

        mname = type(mesh).__name__
        self.dof = self.DOF[spacetype][mname](mesh, p)
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.cellmeasure = mesh.entity_measure('cell')
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        #self.integrator = TetrahedronQuadrature(p+4)
    def interpolation_points(self):
        return self.dof.interpolation_points()

    @barycentric
    def basis(self, bc, index=np.s_[:], p=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(1, ldof)` or `(NQ, 1, ldof)`

        See Also
        --------

        Notes
        -----

        """
        if p is None:
            p = self.p

        NQ = bc.shape[0]
        TD = bc.shape[1]-1
        multiIndex = self.mesh.multi_index_matrix(p, etype=TD)
        ldof = multiIndex.shape[0]

        B = bc
        B = np.ones((NQ, p+1, TD+1), dtype=np.float64)
        B[:, 1:] = bc[:, None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P).reshape(1, -1, 1)
        B = B/P

        # B : (NQ, p+1, TD+1) 
        # B[:, multiIndex, np.arange(TD+1).reshape(1, -1)]: (NQ, ldof, TD+1)
        phi = P[0, -1, 0]*np.prod(B[:, multiIndex, 
            np.arange(TD+1).reshape(1, -1)], axis=-1)
        return phi[..., None, :] 

    @barycentric
    def grad_basis(self, bc, index=np.s_[:], p=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`

        Returns
        -------
        gphi : numpy.ndarray
            the shape of `gphi` can b `(NC, ldof, GD)' or
            `(NQ, NC, ldof, GD)'

        See also
        --------

        Notes
        -----

        """
        if p==None:
            p = self.p

        NQ = bc.shape[0]
        TD = bc.shape[1]-1
        multiIndex = self.mesh.multi_index_matrix(p, TD)
        ldof = multiIndex.shape[0]

        B = bc
        B = np.ones((NQ, p+1, TD+1), dtype=self.ftype)
        B[:, 1:] = bc[:, None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P).reshape(1, -1, 1)
        B = B/P

        F = np.zeros(B.shape, dtype=np.float64)
        F[:, 1:] = B[:, :-1]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            idx = np.array(idx, dtype=np.int_)
            R[..., i] = np.prod(B[..., multiIndex[:, idx], idx.reshape(1, -1)],
                    axis=-1)*F[..., multiIndex[:, i], [i]]

        Dlambda = self.mesh.grad_lambda()
        gphi = P[0, -1, 0]*np.einsum("qlm, cmd->qcld", R, Dlambda, optimize=True)
        return gphi[:, index]
    def mass_matrix(self):
        # TODO 优化效率, Bernstein 基的质量矩阵每个单元是相同的
        import ipdb
        ipdb.set_trace()
        p = self.p
        gdof = self.dof.number_of_global_dofs()
        cell2dof = self.dof.cell_to_dof()
        
        integrator = self.integrator
        bcs, ws = integrator.get_quadrature_points_and_weights()#p=5 order=9 NQ=19

        cm = self.mesh.entity_measure("cell")
        phi = self.basis(bcs)
        
        M = np.einsum('qcl, qcm, q, c->clm', phi, phi, ws, cm, optimize=True)

        I = np.broadcast_to(cell2dof[:, :, None], M.shape) 
        J = np.broadcast_to(cell2dof[:, None, :], M.shape) 
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof),
                       dtype=np.float64)
        return M

    def grad_m_matrix(self, m):
        p = self.p
        gdof = self.dof.number_of_global_dofs()
        cell2dof = self.dof.cell_to_dof()
        integrator = self.integrator #p+4
        bcs, ws = integrator.get_quadrature_points_and_weights()#p=5 order=9 NQ=19

        GD = self.mesh.geo_dimension()
        idx = self.mesh.multi_index_matrix(m, GD-1)
        num = factorial(m)/np.prod(factorial(idx), axis=1)
        #|m|/m！m是相加为m的多重数组,两个 T_1^m

        cm = self.mesh.entity_measure("cell")
        gmphi = self.grad_m_basis(bcs, m)
        M = np.einsum('qclg, qcmg, g, q, c->clm', gmphi, gmphi, num, ws, cm, optimize=True)

        I = np.broadcast_to(cell2dof[:, :, None], M.shape) 
        J = np.broadcast_to(cell2dof[:, None, :], M.shape) 
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof),
                       dtype=np.floa64)
        return M


    def partial_matrix_dense(self):
        """
        @brief 求导矩阵, Bernstein 多项式求导后继续使用 Bernstein 多项式表示
        """
        p = self.p
        mesh = self.mesh
        GD = self.mesh.geo_dimension()
        NC = self.mesh.number_of_cells()
        midxp_0 = mesh.multi_index_matrix(p  , GD) # p   次多重指标
        midxp_1 = mesh.multi_index_matrix(p-1, GD) # p-1 次多重指标
        n0, n1 = len(midxp_0), len(midxp_1)

        if GD==2:
            midx2num = lambda a : (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2 + a[:, 2]
        elif GD==3:
            midx2num = lambda a : (a[:, 1]+a[:, 2]+a[:, 3])*(1+a[:, 1]+a[:,
                2]+a[:, 3])*(2+a[:, 1]+a[:, 2]+a[:, 3])//6 + (a[:, 2]+a[:,
                    3])*(a[:, 2]+a[:, 3]+1)//2 + a[:, 3]

        P = np.zeros((NC, n0, n1, GD), dtype=np.float64)
        involve = np.zeros((n1, n0), dtype=np.float64)
        for i in range(GD+1):
            midxp_1[:, i] += 1
            idx = midx2num(midxp_1)
            involve[np.arange(n1), idx] = midxp_1[:, i]
            midxp_1[:, i] -= 1

        glambda = self.mesh.grad_lambda() #(NC, 4, 3)
        for i in range(GD+1):
            midxp_0[:, i] -= 1
            flag = midxp_0[:, i] >= 0
            idx = midx2num(midxp_0[flag])
            P[:, np.arange(n0)[flag], idx] = glambda[:, i, None, :]
            midxp_0[:, i] += 1

        P = np.transpose(P, (3, 0, 1, 2))
        P = np.einsum('gcij, jk->gcik', P, involve, optimize=True)
        return P

    def partial_matrix(self):
        """
        @brief 求导矩阵, Bernstein 多项式求导后继续使用 Bernstein 多项式表示
               矩阵使用 coo 矩阵表示
        """
        p = self.p
        mesh = self.mesh
        GD = self.mesh.geo_dimension()
        NC = self.mesh.number_of_cells()
        midxp_0 = mesh.multi_index_matrix(p  , GD) # p   次多重指标
        midxp_1 = mesh.multi_index_matrix(p-1, GD) # p-1 次多重指标
        n0, n1 = len(midxp_0), len(midxp_1)

        if GD==2:
            midx2num = lambda a : (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2 + a[:, 2]
        elif GD==3:
            midx2num = lambda a : (a[:, 1]+a[:, 2]+a[:, 3])*(1+a[:, 1]+a[:,
                2]+a[:, 3])*(2+a[:, 1]+a[:, 2]+a[:, 3])//6 + (a[:, 2]+a[:,
                    3])*(a[:, 2]+a[:, 3]+1)//2 + a[:, 3]

        P = np.zeros((NC, n0, n1, GD), dtype=np.float64)
        N = len(midxp_1)
        I = np.zeros((GD+1)*N, dtype=np.int_)
        J = np.zeros((GD+1)*N, dtype=np.int_)
        data = np.zeros((GD+1)*N, dtype=np.float64)
        for i in range(GD+1):
            midxp_1[:, i] += 1
            I[i*N:(i+1)*N] = np.arange(n1)
            J[i*N:(i+1)*N] = midx2num(midxp_1) 
            data[i*N:(i+1)*N] = midxp_1[:, i] 
            midxp_1[:, i] -= 1
        involve = csc_matrix((data, (I, J)), shape=(n1, n0), dtype=np.float64)

        glambda = self.mesh.grad_lambda() #(NC, 4, 3)
        I = [np.array([], dtype=np.int_) for i in range(GD)]
        J = [np.array([], dtype=np.int_) for i in range(GD)]
        data = [np.array([], dtype=np.float64) for i in range(GD)]
        for i in range(GD+1):
            midxp_0[:, i] -= 1
            flag = midxp_0[:, i] >= 0
            idx = midx2num(midxp_0[flag])
            for j in range(GD):
                J[j] = np.r_[J[j], np.tile(idx, (NC,))]
                I[j] = np.r_[I[j], (np.arange(NC)[:, None]*n0 + np.arange(n0)[None, flag]).reshape(-1)]
                data[j] = np.r_[data[j], np.tile(glambda[:, i, j, None], (len(idx), )).reshape(-1)]
            midxp_0[:, i] += 1

        P = []
        for j in range(GD):
            tem = csr_matrix((data[j], (I[j], J[j])), shape = (NC*n0, n1),
                             dtype=np.float64)
            tem = (tem@involve).tocoo()
            II, JJ, ddata = tem.row, tem.col, tem.data
            JJ = JJ + n0*(II//n0)
            M = csr_matrix((ddata, (II, JJ)), shape=(NC*n0, NC*n0),
                           dtype=np.float64)
            P.append(M)
        return P

    def grad_m_basis_0(self, bcs, m):
        """!
        @brief m=3时导数排列顺序: [xxx, yxx, yxy, yyy]
               导数按顺序每个对应一个 A_d^m 的多重指标，对应 alpha 的导数有
               m!/alpha! 个.
        """
        mesh = self.mesh
        phi = self.basis(bcs) # (NQ, 1, ldof)
        if m==0: return phi # 函数值
        phi = phi[:, 0] # 去掉单元轴更方便

        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        P = self.partial_matrix()
        f = lambda x: np.array([int(ss) for ss in '0'*(m-len(np.base_repr(x, GD)))+np.base_repr(x, GD) ], dtype=np.int_)
        idx = np.array(list(map(f, np.arange(GD**m))))

        flag = np.ones(len(idx), dtype=np.bool_)
        for i in range(m-1):
            flag = flag & (idx[:, i]>=idx[:, i+1])
        idx = idx[flag]
        N = len(idx)

        ldof = self.dof.number_of_local_dofs('cell')
        gmphi = np.zeros(phi.shape[:1]+(NC, ldof, N), dtype=self.ftype)
        for i in range(N):
            M = P[idx[i, 0]].copy()
            for j in range(1, m):
                M = M@P[idx[i, j]]
            M = M.tocoo()
            I, J, data = M.row, M.col, M.data
            J = J - ldof*(I//ldof)
            M = csr_matrix((data, (I, J)), shape=(NC*ldof, ldof),
                           dtype=np.float64)
            #gmphi[..., i] = M.dot(phi.T).reshape(NC, ldof, -1)
            for q in range(phi.shape[0]):
                gmphi[q, ..., i] = (M@phi[q]).reshape(NC, -1)
        return gmphi

    def grad_m_basis(self, bcs, m, index=np.s_[:]):
        """
        @brief m=3时导数排列顺序: [xxx, yxx, yxy, yyy]
               导数按顺序每个对应一个 A_d^m 的多重指标，对应 alpha 的导数有
               m!/alpha! 个.
        """
        p = self.p
        p0 = p-m
        if p0<0:
            return np.zeros([1, 1, 1, 1], dtype=np.float64)

        phi = self.basis(bcs, p=p0)
        NQ = bcs.shape[0]

        mesh = self.mesh
        if m==0: return phi # 函数值
        phi = phi[:, 0] # 去掉单元轴更方便

        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs('cell')
        glambda = mesh.grad_lambda()

        ## 获得张量对称部分的索引
        symidx = symmetry_index(GD, m)

        ## 计算多重指标编号
        if GD==2:
            midx2num = lambda a : (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2 + a[:, 2]
        elif GD==3:
            midx2num = lambda a : (a[:, 1]+a[:, 2]+a[:, 3])*(1+a[:, 1]+a[:,
                2]+a[:, 3])*(2+a[:, 1]+a[:, 2]+a[:, 3])//6 + (a[:, 2]+a[:,
                    3])*(a[:, 2]+a[:, 3]+1)//2 + a[:, 3]

        midxp_0 = mesh.multi_index_matrix(p, GD) # p   次多重指标
        midxp_1 = mesh.multi_index_matrix(m, GD) # m   次多重指标

        N, N1 = len(symidx), midxp_1.shape[0]
        B = np.zeros((N1, NQ, ldof), dtype=np.float64)
        symLambdaBeta = np.zeros((N1, NC, N), dtype=np.float64)
        gmphi = np.zeros((NQ, ldof, NC, N), dtype=np.float64)
        for beta, Bi, symi in zip(midxp_1, B, symLambdaBeta):
            midxp_0 -= beta[None, :]
            idx = np.where(np.all(midxp_0>-1, axis=1))[0]
            num = midx2num(midxp_0[idx]) 
            symi[:] = symmetry_span_array(glambda, beta).reshape(NC, -1)[:, symidx] #(NC, N)
            c = (factorial(m)**2)*comb(p, m)/np.prod(factorial(beta)) # 数
            Bi[:, idx] = c*phi[:, num] #(NQ, ldof)
            midxp_0 += beta[None, :]
        gmphi = np.einsum('iql, icn->qcln', B, symLambdaBeta[:, index], optimize=True)
        return gmphi

    def hess_basis(self, bcs, index=np.s_[:]):
        if self.p<2:
            return np.zeros([1, 1, 1, 1], dtype=np.float64)
        g2phi = self.grad_m_basis(bcs, 2, index=index)
        TD = self.mesh.top_dimension()
        shape = g2phi.shape[:-1] + (TD, TD)
        hval  = np.zeros(shape, dtype=np.float64)
        hval[..., 0, 0] = g2phi[..., 0]
        hval[..., 0, 1] = g2phi[..., 1]
        hval[..., 1, 0] = g2phi[..., 1]
        hval[..., 1, 1] = g2phi[..., 2]
        return hval

    def lagrange_to_bernstein(self, p = 1, TD = 1):
        '''
        @brief 将 Bernstein 基函数转换为 lagrange 基函数。即 b_i = l_j A_{ji}
            其中 b_i 为 Bernstein 基函数，l_i 为 lagrange 基函数.
        '''
        bcs = self.mesh.multi_index_matrix(p, TD)/p # p   次多重指标
        return self.basis(bcs, p=p)[:, 0]
        

    def bernstein_to_lagrange(self, p=1, TD=1):
        '''
        @brief 将 Bernstein 基函数转换为 lagrange 基函数。即 l_i = b_j A_{ji}
            其中 b_i 为 Bernstein 基函数，l_i 为 lagrange 基函数.
        '''
        return np.linalg.inv(self.lagrange_to_bernstein(p, TD))
        

    @barycentric
    def value(self, 
            uh: np.ndarray, 
            bc: np.ndarray, 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
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
        gdof = self.dof.number_of_global_dofs()
        phi = self.basis(bc, index=index) # (NQ, NC, ldof)
        cell2dof = self.dof.cell_to_dof(index=index)

        dim = len(uh.shape) - 1
        s0 = 'abdefg'
        if self.doforder == 'sdofs':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., NC)
            s1 = f"...ci, {s0[:dim]}ci->...{s0[:dim]}c"
            val = np.einsum(s1, phi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ...)
            s1 = f"...ci, ci{s0[:dim]}->...c{s0[:dim]}"
            val = np.einsum(s1, phi, uh[cell2dof, ...])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'sdofs' and 'vdims'.")
        return val


    @barycentric
    def grad_value(self, 
            uh: np.ndarray, 
            bc: np.ndarray, 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        """
        @note
        """
        gdof = self.dof.number_of_global_dofs()
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell_to_dof(index=index)
        dim = len(uh.shape) - 1
        s0 = 'abdefg'

        if dim == 0: # 如果
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (gdof, )
            # uh[cell2dof].shape == (NC, ldof)
            # val.shape == (NQ, NC, GD)
            val = np.einsum('...cim, ci->...cm', gphi, uh[cell2dof[index]])
        elif self.doforder == 'sdofs':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., GD, NC)
            s1 = '...cim, {}ci->...{}mc'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ..., GD)
            s1 = '...cim, ci{}->...c{}m'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[cell2dof[index], ...])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'sdofs' and 'vdims'.")

        return val

    @barycentric
    def grad_m_value(self, uh, bcs, m):
        gmphi = self.grad_m_basis(bcs, m) # (NQ, 1, ldof)
        cell2dof = self.dof.cell_to_dof()
        val = np.einsum('qclg, cl->qcg', gmphi, uh[cell2dof])
        return val

    @barycentric
    def hessian_value(self, 
            uh: np.ndarray, 
            bc: np.ndarray, 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        """
        @note
        """
        gdof = self.dof.number_of_global_dofs()
        gphi = self.hess_basis(bc, index=index)
        cell2dof = self.dof.cell_to_dof(index=index)
        dim = len(uh.shape) - 1
        s0 = 'abdefg'
        val = np.einsum('...cimn, ci->...cmn', gphi, uh[cell2dof[index]])
        return val

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, 
                coordtype='barycentric', dtype=dtype)

    def array(self, dim=None, dtype=np.float64):
        gdof = self.dof.number_of_global_dofs()
        if dim is None:
            dim = tuple() 
        if type(dim) is int:
            dim = (dim, )

        if self.doforder == 'sdofs':
            shape = dim + (gdof, )
        elif self.doforder == 'vdims':
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

    def interpolate(self, u, dim=None, dtype=None):
        """
        @brief Interpolates a function `u` in the finite element space.
        """
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

        l2b = self.bernstein_to_lagrange(self.p, self.TD)
        c2d = self.dof.cell2dof
        uI[c2d] = np.einsum('ij, cj->ci', l2b, uI[c2d])
        return self.function(dim=dim, array=uI, dtype=uI.dtype)
















from typing import Union, TypeVar, Generic, Callable, Optional

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from ..decorator import barycentric
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof
from .functional import*
from scipy.special import factorial, comb

_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)


class BernsteinFESpace(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p
        self.device = mesh.device

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self) -> int:
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        return self.dof.interpolation_points()

    def cell_to_dof(self, index: Index = _S) -> TensorLike:
        return self.dof.cell_to_dof(index=index)

    def face_to_dof(self, index: Index = _S) -> TensorLike:
        return self.dof.face_to_dof(index=index)

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold, method=method)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")


    @barycentric
    def basis(self, bcs: TensorLike, index: Index=_S, p = None):
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

        NQ = bcs.shape[0]
        TD = bcs.shape[1]-1
        multiIndex = self.mesh.multi_index_matrix(p, etype=TD)
        ldof = multiIndex.shape[0]

        B = bcs
        B = bm.ones((NQ, p+1, TD+1), device=self.device, dtype=self.ftype)
        # B[:, 1:] = bcs[:, None, :]
        B = bm.set_at(B,(slice(None),slice(1,None)),bcs[:,None,:])
        B = bm.cumprod(B, axis=1)
        P = bm.arange(p+1, device=self.device)
        P = bm.set_at(P,(0),1)
        P = bm.cumprod(P,axis=0).reshape(1, -1, 1)
        B = B/P

        # B : (NQ, p+1, TD+1) 
        # B[:, multiIndex, bm.arange(TD+1).reshape(1, -1)]: (NQ, ldof, TD+1)
        phi = P[0, -1, 0]*bm.prod(B[:, multiIndex, bm.arange(TD+1).reshape(1, -1)], axis=-1)
        return phi[None, :]

    @barycentric
    def grad_basis(self, bcs: TensorLike, index: Index=_S, variable='x',p=None):
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

        NQ = bcs.shape[0]
        TD = bcs.shape[1]-1
        multiIndex = self.mesh.multi_index_matrix(p, TD)
        ldof = multiIndex.shape[0]

        B = bcs
        B = bm.ones((NQ, p+1, TD+1), device=self.device, dtype=self.ftype)
        # B[:, 1:] = bcs[:, None, :]
        B = bm.set_at(B,(slice(None),slice(1,None)),bcs[:,None,:])
        B = bm.cumprod(B, axis=1)

        P = bm.arange(p+1,device=self.device)
        # P[0] = 1
        P = bm.set_at(P,(0),1)
        P = bm.cumprod(P,axis=0).reshape(1, -1, 1)
        B = B/P

        F = bm.zeros(B.shape, device=self.device, dtype=self.ftype)
        # F[:, 1:] = B[:, :-1]
        F = bm.set_at(F,(slice(None),slice(1,None)),B[:,:-1])
        shape = bcs.shape[:-1]+(ldof, TD+1)
        R = bm.zeros(shape,device=self.device,dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            idx = bm.array(idx,device=self.device, dtype=self.itype)
            # R[..., i] = bm.prod(B[..., multiIndex[:, idx], idx.reshape(1, -1)],axis=-1)*F[..., multiIndex[:, i], [i]]
            R = bm.set_at(R,(...,i),bm.prod(B[..., multiIndex[:, idx], idx.reshape(1, -1)],axis=-1)*F[..., multiIndex[:, i], [i]])
        if variable == 'lambda':
            return P[0, -1, 0]*R
        elif variable == 'x':
            Dlambda = self.mesh.grad_lambda()
            gphi = P[0, -1, 0]*bm.einsum("qlm, cmd->cqld", R, Dlambda)# TODO: optimize
            return gphi[:, index]

    @barycentric
    def hess_basis(self, bcs: TensorLike, index: Index=_S, variable='u'):
        """
        @brief Compute the Hessian of the basis function values at barycentric 
               point bc
        """
        g2phi = self.grad_m_basis(bcs, 2, index=index)
        TD = self.mesh.top_dimension()
        shape = g2phi.shape[:-1] + (TD, TD)
        hval  = bm.zeros(shape, device=self.device, dtype=self.ftype)
        hval = bm.set_at(hval,(...,0,0),g2phi[..., 0])
        hval = bm.set_at(hval,(...,0,1),g2phi[..., 1])
        hval = bm.set_at(hval,(...,1,0),g2phi[..., 1])
        hval = bm.set_at(hval,(...,1,1),g2phi[..., 2])
        return hval

    @barycentric
    def grad_m_basis(self, bcs: TensorLike, m: int, index = _S):
        """
        @brief Compute the m-th order gradient of the basis function values at
               the barycentric point `bc`. The gradient is a GD-dim and m-th
               order sysmmetry tensor with shape (NQ, NC, ldof, N), where N is
               the number of the gradients.  
        @return TensorLike with shape (NQ, NC, ldof, N)
               Where N is the number of the gradients, which is equal to the 
               number of GD-dim and m-th order symmetry tensor.
               For example, in the case of m = 3 and GD = 2, the order of gradient
               is [xxx, xxy, xyy, yyy], and the shape of the output is 
               (NQ, NC, ldof, 4), where 4 is the number of the gradients.
               Additionally, 
               时导数排列顺序: [xxx, xxy, xyy, yyy]
               导数按顺序每个对应一个 A_d^m 的多重指标，对应 alpha 的导数有
               m!/alpha! 个.
        """
        p = self.p
        mesh = self.mesh
        if(p - m <0): return bm.zeros([1, 1, 1, 1], dtype=self.ftype)

        phi = self.basis(bcs, p=p-m)
        NQ = bcs.shape[0]

        if m==0: return phi # 函数值
        phi = phi[0] # 去掉单元轴更方便 (NQ, ldof)

        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs('cell')
        glambda = mesh.grad_lambda()

        ## 获得张量对称部分的索引
        symidx,_ = symmetry_index(GD, m)

        ## 计算多重指标编号
        if GD==2:
            midx2num = lambda a : (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2 + a[:, 2]
        elif GD==3:
            midx2num = lambda a : (a[:, 1]+a[:, 2]+a[:, 3])*(1+a[:, 1]+a[:,
                2]+a[:, 3])*(2+a[:, 1]+a[:, 2]+a[:, 3])//6 + (a[:, 2]+a[:,
                    3])*(a[:, 2]+a[:, 3]+1)//2 + a[:, 3]

        midxp_0 = bm.multi_index_matrix(p, GD) # p   次多重指标
        midxp_1 = bm.multi_index_matrix(m, GD) # m   次多重指标

        N, N1 = len(symidx), midxp_1.shape[0]
        B = bm.zeros((N1, NQ, ldof), device=self.device, dtype=self.ftype)
        symLambdaBeta = bm.zeros((N1, NC, N), dtype=self.ftype, device=self.device)
        for beta, Bi, symi in zip(midxp_1, B, symLambdaBeta):
            midxp_0 -= beta[None, :]
            idx = bm.where(bm.all(midxp_0>-1, axis=1))[0]
            num = midx2num(midxp_0[idx]) 
            beta = bm.to_numpy(beta)
            fbeta = bm.tensor(factorial(beta))
            beta = bm.array(beta)
            symi = bm.set_at(symi,(slice(None)),symmetry_span_array(glambda, beta).reshape(NC, -1)[:, symidx])
            c = (factorial(m)**2)*comb(p, m)/bm.prod(fbeta,axis=0,dtype=self.itype) # 数
            Bi = bm.set_at(Bi,(slice(None),idx),c*phi[:, num])
            midxp_0 += beta[None, :]
        gmphi = bm.einsum('iql, icn->cqln', B, symLambdaBeta[:, index])
        return gmphi

    @barycentric
    def value(self, uh: TensorLike, bcs: TensorLike, index: Index=_S) -> TensorLike:
        """
        """
        phi = self.basis(bcs, index=index)
        TD = bcs.shape[-1] - 1
        e2d = self.dof.entity_to_dof(etype=TD, index=index)
        val = bm.einsum('cql, ...cl -> ...cq', phi, uh[..., e2d])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bcs: TensorLike, index: Index=_S) -> TensorLike:
        """
        @brief
        """
        gphi = self.grad_basis(bcs, index=index)
        cell2dof = self.dof.cell_to_dof(index=index)
        val = bm.einsum('cqlm, ...cl->...cqm', gphi, uh[..., cell2dof])
        return val
    
    @barycentric
    def grad_m_value(self, uh, bcs, m):
        gmphi = self.grad_m_basis(bcs, m) # (NQ, 1, ldof)
        cell2dof = self.dof.cell_to_dof()
        val = bm.einsum('cqlg, cl->cqg', gmphi, uh[cell2dof])
        return val

    @barycentric
    def hessian_value(self, 
            uh: TensorLike, 
            bcs: TensorLike, 
            index: Union[TensorLike, slice]=_S
            ) -> TensorLike:
        """
        @note
        """
        gdof = self.dof.number_of_global_dofs()
        hphi = self.hess_basis(bcs, index=index)
        cell2dof = self.dof.cell_to_dof(index=index)
        val = bm.einsum('cqlij, cl->cqij', hphi, uh[cell2dof[index]])
        return val    

    def lagrange_to_bernstein(self, p = 1, TD = 1):
        '''
        @brief Convert Lagrange basis functions to Bernstein basis functions. That is,
                                b_i = l_j A_{ji}
            where b_i is the Bernstein basis function, l_i is the Lagrange basis
            function.  Conversely, A converts Bernstein coefficients to Lagrange
            coefficients.
        '''
        bcs = self.mesh.multi_index_matrix(p, TD, dtype=self.ftype)/p # p   次多重指标
        return self.basis(bcs, p=p)[0]


    def bernstein_to_lagrange(self, p=1, TD=1):
        '''
        @brief Convert Bernstein basis functions to Lagrange basis functions. That is,
                                l_i = b_j A_{ji}
            where b_i is the Bernstein basis function, l_i is the Lagrange basis
            function.  Conversely, A converts Lagrange coefficients to Bernstein
            coefficients.
        '''
        return bm.linalg.inv(self.lagrange_to_bernstein(p, TD))
    

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
        c2d = self.cell_to_dof()
        #uI[c2d] = bm.einsum('ij, cj->ci', l2b, uI[c2d])
        uI = bm.set_at(uI,(c2d),bm.einsum('ij, cj->ci', l2b, uI[c2d]))
        return uI

    def boundary_interpolate(self, gd: Union[Callable, int, float],
            uh: TensorLike, threshold=None, method=None) -> TensorLike:
        """
        @brief Interpolates the Dirichlet boundary condition.
        """

        gdof = self.number_of_global_dofs()
        isbdface = self.mesh.boundary_face_flag()
        f2d  = self.face_to_dof()[isbdface]

        isDDof = bm.zeros(gdof, device=self.device, dtype=bm.bool)
        # isDDof[f2d] = True
        isDDof = bm.set_at(isDDof,(f2d),True)
        if callable(gd):
            ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
            gd = gd(ipoints[isDDof])

        # uh[isDDof] = gD
        uh = bm.set_at(uh,(isDDof), gd)
        l2b = self.bernstein_to_lagrange(self.p, self.TD - 1) 
        # uh[f2d] = bm.einsum('ij, fj->fi', l2b, uh[f2d])
        uh = bm.set_at(uh,(f2d),bm.einsum('ij, fj->fi', l2b, uh[f2d])) 
        return self.function(uh), isDDof

    set_dirichlet_bc = boundary_interpolate






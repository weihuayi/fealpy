from typing import Optional
from scipy.special import factorial, comb
from fealpy.functionspace.functional import symmetry_index, symmetry_span_array

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
)

class MLaplaceBernsteinIntegrator(LinearInt, OpInt, CellInt):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    """
    m 求导次数
    """
    def __init__(self, m: int=2,coef: Optional[CoefLike]=None, q:
            Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.m = m
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        m = self.m
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The gradmIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__}is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gmphi = space.grad_m_basis(bcs, m=m)
        return bcs, ws, gmphi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        """
        @parm    space: berstein space
        """
        m = self.m
        mesh = getattr(space, 'mesh', None)
        p = space.p

        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        cellmeasure = mesh.entity_measure('cell')
        multiIndex = mesh.multi_index_matrix(p-m, GD)
        mmultiIndex = mesh.multi_index_matrix(m, GD)
        pmultiIndex = mesh.multi_index_matrix(p, GD)
        l = multiIndex.shape[0] # 矩阵的行数
        B = bm.zeros((l, l), dtype=space.ftype)

        # 积分公式
        def integrator(a): # a 为多重指标
            value = factorial(int(GD))*bm.prod(bm.array(factorial(bm.array(a))))/factorial(int(bm.sum(a)+GD)) 
            return value * cellmeasure[0]

        # 计算B^{\alpha-beta}矩阵
        for i in range(l):
            for j in range(l):
                alpha = multiIndex[i] + multiIndex[j]
                c = (1/bm.prod(bm.array(factorial(multiIndex[i]))))*(1/bm.prod(bm.array(factorial(multiIndex[j]))))
                B[i, j] = c * integrator(alpha)

        gmB = bm.zeros((NC, pmultiIndex.shape[0], pmultiIndex.shape[0]), dtype=space.ftype) 
        glambda = mesh.grad_lambda()
        for beta in mmultiIndex:
            idx1 = bm.where(bm.all(pmultiIndex-beta[None,:] >= 0, axis=1))[0]
            multiidx1 = mesh.multi_index_matrix(bm.sum(beta), GD-1)
            symidx, num = symmetry_index(GD, bm.sum(beta)) 

            # \Lambda^{\beta}
            glambda1 = symmetry_span_array(glambda, beta).reshape(NC, -1)[:, symidx]
            idx11 = bm.broadcast_to(idx1[:, None], (l, l))
            c1 = factorial(int(bm.sum(beta)))/bm.prod(bm.array(factorial(beta)))
            for theta in mmultiIndex:
                idx2 = bm.where(bm.all(pmultiIndex-theta[None,:] >= 0, axis=1))[0]
                glambda2 = symmetry_span_array(glambda, theta).reshape(NC, -1)[:, symidx]
                c2 = factorial(int(bm.sum(theta)))/bm.prod(bm.array(factorial(theta)))

                # \Lambda^{\beta}: \Lambda^{\theta}
                symglambda = bm.einsum('ci,ci,i->c', glambda1, glambda2, num)
                idx22 = bm.broadcast_to(idx2[None, :], (l, l))
                B1 = B * c1 * c2
                #BB = B1[None, :, :] * symglambda[:, None, None]
                BB = bm.multiply(B1, symglambda[:, None, None])
                #bm.add_at(gmB, (slice(None), idx11, idx22), B1[None, :, :] * symglambda[:, None, None])
                #bm.add_at(gmB, (slice(None), idx11, idx22), BB)
                #gmB[:, idx11, idx22] += B1[None, :, :] * symglambda[:, None, None]
                gmB[:, idx11, idx22] += BB
                del B1,BB 
                import gc
                gc.collect()
        del B
        gc.collect()
        gmB = gmB * factorial(int(p)) * factorial(int(p))
        return gmB

                






















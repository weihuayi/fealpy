from typing import Optional
from scipy.special import factorial

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod,
    CoefLike
)


class MthLaplaceIntegrator(LinearInt, OpInt, CellInt):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    def __init__(self, m: int=2,coef: Optional[CoefLike]=None, q:
            Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
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
        m = self.m
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        device = bm.get_device(mesh.cell)
        GD = mesh.geo_dimension()
        idx = bm.array(mesh.multi_index_matrix(m, GD-1))
        idx_cpu = bm.to_numpy(idx)
        num = factorial(m)/bm.prod(bm.array(factorial(idx_cpu), device=device), axis=1)
        bcs, ws, gmphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell',
                                 index=index)
        M = bm.einsum('cqlg,cqmg,g,q,c->clm',gmphi,gmphi,num,ws,cm)
        return M
    #return bilinear_integral(gmphi1, gmphi, ws, cm, coef,
    #                             batched=self.batched)

    @assemblymethod('without_numerical_integration')
    def assembly_without_numerical_integration(self, space: _FS):
        from .bilinear_form import BilinearForm
        from .mlaplace_bernstein_integrator import MLaplaceBernsteinIntegrator
        m = self.m
        q = space.p+3 if self.q is None else self.q
        cmcoeff = space.coeff
        bgm = MLaplaceBernsteinIntegrator(m=m, q=q).assembly(space.bspace)
        #M = bm.einsum('cil,clm,cpm->cip', cmcoeff, bgm, cmcoeff)
        M = cmcoeff @ bgm @ cmcoeff.transpose(0,2,1)
        #import numpy as np
        #np.save('Mcn8.npy', M)
        return M

        

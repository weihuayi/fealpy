from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..utils import is_tensor

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from ..decorator.variantmethod import variantmethod
from ..decorator import cartesian
from .integrator import LinearInt, OpInt, CellInt, enable_cache

class SpaceTimeResidualIntegrator(LinearInt, OpInt, CellInt):
    """
    The residual integrator for function spaces based on homogeneous meshes.
    This class integrates the residual term in space-time finite element methods.
    It supports both constant and variable residual coefficients.
    """
    def __init__(self, pde, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 theta: float = 0.5,
                 epsilon: float = 0,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.pde = pde
        self.diff_coef = pde.diffusion_coef
        self.conv_coef = pde.convection_coef
        self.reac_coef = pde.reaction_coef
        self.q = q
        self.index = index
        self.batched = batched
        self.theta = theta
        self.epsilon = epsilon
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The SpaceTimeConvectionIntegrator only supports spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomogeneousMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p + 3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        gphi = space.grad_basis(bcs, index=index)
        hphi = space.hess_basis(bcs, index=index)
        return bcs, ws, phi, gphi, hphi, cm, index

    @cartesian
    def diff_coef_expand(self, p: TensorLike, index: Index = _S) -> TensorLike:
        """
        Expand the diffusion coefficient to match the shape of the basis function gradients.

        Parameters:
            coef (TensorLike): The diffusion coefficients at quadrature points.
        """
        epsilon = self.epsilon
        if not callable(self.diff_coef):
            coef = self.diff_coef
        else:
            coef = self.diff_coef(p)
        shape = p.shape + (p.shape[-1],)
        dim = p.shape[-1]
        idx = bm.arange(dim-1)
        coef_exp = bm.zeros(shape, dtype=bm.float64)
        coef_exp = bm.set_at(coef_exp, (...,idx,idx), coef)
        coef_exp = bm.set_at(coef_exp, (...,-1,-1), epsilon)
        return coef_exp[index]
    
    @cartesian
    def conv_coef_expand(self, p: TensorLike, index: Index = _S) -> TensorLike:
        """
        Expand the convection coefficient to match the shape of the basis function gradients.
        
        Parameters:
            coef (TensorLike): The convection coefficients at quadrature points.
        """
        if not callable(self.conv_coef):
            coef = self.conv_coef
        else:
            coef = self.conv_coef(p)
        shape = p.shape
        if isinstance(coef, (int, float)):
            coef_exp = bm.broadcast_to(coef, shape).copy()
            coef_exp = bm.set_at(coef_exp, (...,-1),1.0)
            return coef_exp[index]
        elif is_tensor(coef):
            if coef.ndim == 1:
                coef = bm.concat([coef , bm.array([1.0], dtype=coef.dtype)], axis=0)
                return bm.broadcast_to(coef, shape)[index]
            elif coef.ndim >= 2:
                ones = bm.ones((coef.shape[:-1]+(1,)), dtype=coef.dtype)
                coef_exp = bm.concat([coef, ones], axis=-1)
                return coef_exp[index]

    def residual_correction(self,space:_FS,cm,gphi,conv_coef, index):
        """
        Apply additional correction to gphi to obtain a new gphi
    
        Parameters:
            p (int): Polynomial order of the finite element space.
            cm (TensorLike): Cell measures.
            phi (TensorLike): Basis functions.
            gphi (TensorLike): Gradient of basis functions.
            coef (TensorLike): Convection coefficients.
        Returns:
            TensorLike: Corrected test functions.
        """
        index = self.index
        p = space.p
        if p == 1:
            m = 1
        elif p >= 2:
            m = 2
        s = 1/bm.sqrt(bm.sum(conv_coef[index]**2, axis=-1))[..., None]
        h = bm.sqrt(cm[index])
        for _ in range(conv_coef.ndim-1):
            h = h[..., None]
        term_correction = bm.einsum('cqi...k, cq...k->cqi...' ,gphi, conv_coef[index])
        residual_phi = self.theta*h**m*s*term_correction
        return residual_phi

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, gphi, hphi, cm, index = self.fetch(space)
        diff_coef = process_coef_func(self.diff_coef_expand, bcs=bcs, mesh=mesh, etype='cell', index=index)
        conv_coef = process_coef_func(self.conv_coef_expand, bcs=bcs, mesh=mesh, etype='cell', index=index)
        reac_coef = process_coef_func(self.reac_coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        diff_phi = bm.einsum('cqi...jk,cq...jk->cqi...' ,-hphi, diff_coef)
        conv_phi = bm.einsum('cqi...j,cq...j->cqi...' ,gphi, conv_coef)
        if not is_tensor(reac_coef):
            reac_phi = reac_coef * phi
        else:
            reac_phi = bm.einsum('cqi...,cq...->cqi...' ,phi, reac_coef)
        residual_phi = self.residual_correction(space,cm,gphi,conv_coef, index=index)
        residual = diff_phi + conv_phi + reac_phi
        result = bilinear_integral(residual, residual_phi, ws, cm, coef=None, batched=self.batched)
        return result
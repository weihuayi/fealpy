from typing import Optional, Literal

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, SourceLike,CoefLike

from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func, is_tensor
from ..functional import linear_integral
from ..decorator.variantmethod import variantmethod
from ..decorator import cartesian
from .integrator import LinearInt, SrcInt, CellInt, enable_cache


class SpaceTimeSourceResidualIntegrator(LinearInt, SrcInt, CellInt):
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, pde, q: int=None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool=False,
                 theta: float = 0.5,
                 method: Literal['isopara', None] = None) -> None:
        super().__init__()
        self.pde = pde
        self.conv_coef = pde.convection_coef
        self.source = pde.source
        self.q = q
        self.set_region(region)
        self.batched = batched
        self.theta = theta
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        if indices is None:
            return space.cell_to_dof()
        return space.cell_to_dof(index=self.entity_selection(indices))

    @enable_cache
    def fetch(self, space: _FS, /, indices=None):
        q = self.q
        index = self.entity_selection(indices)
        mesh = getattr(space, 'mesh', None)

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index)

        return bcs, ws, gphi, cm, index

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
        
    def residual_correction(self,space:_FS,cm,gphi,conv_coef):
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
        p = space.p
        if p == 1:
            m = 1
        elif p >= 2:
            m = 2
        s = 1/bm.sqrt(bm.sum(conv_coef**2, axis=-1))[..., None]
        h = bm.sqrt(cm)
        for _ in range(conv_coef.ndim-1):
            h = h[..., None]
        term_correction = bm.einsum('cqi...k, cq...k->cqi...' ,gphi, conv_coef)
        residual_phi = self.theta*h**m*s*term_correction
        return residual_phi
    
    @variantmethod
    def assembly(self, space: _FS, indices=None) -> TensorLike:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space, indices)
        ps = mesh.bc_to_point(bcs)
        conv_coef = self.conv_coef_expand(ps, index=index)
        residual_phi = self.residual_correction(space, cm, gphi, conv_coef)
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return linear_integral(residual_phi, ws, cm, val, batched=self.batched)
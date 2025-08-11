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

class SpaceTimeDiffusionIntegrator(LinearInt, OpInt, CellInt):
    """
    The diffusion integrator for function spaces based on homogeneous meshes.
    This class integrates the diffusion term in space-time finite element methods.
    It supports both constant and variable diffusion coefficients.
    """
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 theta: float = 1e-5,
                 epsilon: float = 1e-12,
                 conv_coef: Optional[CoefLike]=None,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
        self.theta = theta
        self.epsilon = epsilon
        self.conv_coef = conv_coef
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
        gphi = space.grad_basis(bcs, index=index)
        return bcs, ws, gphi, cm, index

    @cartesian
    def diff_coef_expand(self, p: TensorLike, index: Index = _S) -> TensorLike:
        """
        Expand the diffusion coefficient to match the shape of the basis function gradients.

        Parameters:
            coef (TensorLike): The diffusion coefficients at quadrature points.
        """
        epsilon = self.epsilon
        if not callable(self.coef):
            coef = self.coef
        else:
            coef = self.coef(p)
        shape = p.shape + (p.shape[-1],)
        dim = p.shape[-1]
        
        coef_exp = bm.zeros(shape, dtype=bm.float64)
        coef_exp = bm.set_at(coef_exp, (...,slice(-dim+1),slice(-dim+1)), coef)
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
                ones = bm.ones((coef.shape[0],1), dtype=coef.dtype)
                coef_exp = bm.concat([coef, ones], axis=-1)
                return coef_exp[index]

    def gphi_correction(self,space:_FS,bcs,cm,gphi,coef):
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
        s = 1/bm.sqrt(bm.sum(coef[index]**2, axis=-1))[..., None, None]
        h = bm.sqrt(cm[index])
        for _ in range(coef.ndim):
            h = h[..., None]
        hphi = space.hess_basis(bc=bcs, index=index)
        term_correction = bm.einsum('cqi...jk, cq...k->cqi...j' ,hphi, coef[index])
        gphi_correction = gphi[index] + self.theta*h**p*s*term_correction
        return gphi_correction
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(self.diff_coef_expand, bcs=bcs, mesh=mesh, etype='cell', index=index)

        if is_tensor(coef):
            result = bilinear_integral(gphi, gphi, ws, cm, coef, batched=self.batched)
        else:
            raise TypeError(f"coef should be Tensor-like object but got {type(coef)}")
        return result
    
    @assembly.register('SUPG')
    def assembly(self, space: _FS) -> TensorLike:
        """
        Special assembly method for SUPG (Streamline Upwind Petrov-Galerkin) stabilization.
        This method is used when the diffusion coefficient is variable and requires stabilization.
        """
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(self.diff_coef_expand, bcs=bcs, mesh=mesh, etype='cell', index=index)
        
        ps = mesh.bc_to_point(bcs, index=index)
        conv_coef = self.conv_coef_expand(ps, index=index)
        cgphi = self.gphi_correction(space,bcs,cm,gphi,conv_coef)
        
        if is_tensor(coef):
            result = bilinear_integral(gphi, cgphi, ws, cm, coef, batched=self.batched)
        else:
            raise TypeError(f"coef should be Tensor-like object but got {type(coef)}")
        return result
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

class SpaceTimeConvectionIntegrator(LinearInt, OpInt, CellInt):
    """
    The convection integrator for function spaces based on homogeneous meshes.
    This class integrates the convection term in space-time finite element methods.
    It supports both constant and variable convection coefficients.
    """
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 theta: float = 5e-2,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
        self.theta = theta
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
        phi = space.basis(bcs, index=index)
        return bcs, ws, phi, gphi, cm, index

    @cartesian
    def conv_coef_expand(self, p: TensorLike, index: Index = _S) -> TensorLike:
        """
        Expand the convection coefficient to match the shape of the basis function gradients.
        
        Parameters:
            coef (TensorLike): The convection coefficients at quadrature points.
        """
        if not callable(self.coef):
            coef = self.coef
        else:
            coef = self.coef(p)
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

    def phi_correction(self,p,cm,phi,gphi,coef):
        """
        Apply additional correction to phi to obtain a new phi
        
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
        s = 1/bm.sqrt(bm.sum(coef[index]**2, axis=-1))[..., None]
        h = bm.sqrt(cm[index])
        for _ in range(coef.ndim - 1):
            h = h[..., None]
        term_correction = bm.einsum('cqi...j, cq...j->cqi...' ,gphi[index], coef[index])
        phi_correction = phi[index] + self.theta*h**p*s*term_correction
        return phi_correction
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(self.conv_coef_expand, bcs=bcs, mesh=mesh, etype='cell', index=index)

        if is_tensor(coef):
            gphi = bm.einsum('cqi...j,cq...j->cqi...' ,gphi, coef)
            result = bilinear_integral(phi, gphi, ws, cm, coef=None, batched=self.batched)
        else:
            raise TypeError(f"coef should be Tensor-like object but got {type(coef)}")
        return result
    
    @assembly.register('SUPG')
    def assembly(self, space: _FS) -> TensorLike:
        """
        Special assembly method for SUPG (Streamline Upwind Petrov-Galerkin) stabilization.
        This method is used when the convection coefficient is variable and requires stabilization.
        """
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(self.conv_coef_expand, bcs=bcs, mesh=mesh, etype='cell', index=index)
        cphi = self.phi_correction(space.p,cm,phi,gphi,coef)
        
        if is_tensor(coef):
            gphi = bm.einsum('cqi...j,cq...j->cqi...' ,gphi, coef)
            result = bilinear_integral(cphi, gphi, ws, cm, coef=None, batched=self.batched)
        else:
            raise TypeError(f"coef should be Tensor-like object but got {type(coef)}")
        return result